"""Watermarking with text or a second image.

Both paths draw onto a transparent RGBA overlay and composite it once, which
keeps opacity uniform, prevents repeated resampling of the base image, and means
the watermark colour never interacts with the source pixels.

Tiled mode exists for a real reason: a single corner watermark is trivially
cropped off, so "proof" and "draft" overlays are usually repeated. Tiling is
bounded by a maximum tile count so a small tile on a large canvas cannot
generate unbounded draw work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PIL import Image, ImageDraw

from ..errors import InvalidArgumentError
from .fonts import FontIndex
from .ops import Size, canvas_mode_for, parse_gravity

logger = logging.getLogger(__name__)

#: Hard ceiling on tiles drawn in tiled mode.
_MAX_TILES = 400

#: Fraction of the image's shorter edge used when ``font_size`` is omitted.
_DEFAULT_FONT_RATIO = 0.045

#: Fraction of the image's shorter edge used for an auto-scaled logo.
_DEFAULT_LOGO_RATIO = 0.18


@dataclass(slots=True)
class TextWatermark:
    text: str
    position: str = "bottom-right"
    opacity: float = 0.6
    font_size: int | None = None
    font_family: str | None = None
    colour: tuple[int, int, int] = (255, 255, 255)
    padding: int = 16
    tile: bool = False
    spacing: int = 0
    rotation: float = 0.0
    shadow: bool = True
    stroke_width: int = 0


@dataclass(slots=True)
class ImageWatermark:
    overlay: Image.Image
    position: str = "bottom-right"
    opacity: float = 0.6
    scale: float | None = None
    padding: int = 16
    tile: bool = False
    spacing: int = 0


def _validate_opacity(opacity: float) -> float:
    if not 0.0 <= opacity <= 1.0:
        raise InvalidArgumentError("watermark opacity must be between 0 and 1")
    return opacity


def _apply_opacity(layer: Image.Image, opacity: float) -> Image.Image:
    """Scale a layer's alpha channel by ``opacity``."""
    if opacity >= 1.0:
        return layer
    alpha = layer.getchannel("A").point(lambda value: int(value * opacity))
    layer.putalpha(alpha)
    return layer


def _gravity_offset(canvas: Size, item: Size, position: str, padding: int) -> tuple[int, int]:
    x_ratio, y_ratio = parse_gravity(position)
    x = round((canvas.width - item.width) * x_ratio)
    y = round((canvas.height - item.height) * y_ratio)
    # Padding only applies on the edges the gravity actually pins to.
    if x_ratio == 1.0:
        x -= padding
    elif x_ratio == 0.0:
        x += padding
    if y_ratio == 1.0:
        y -= padding
    elif y_ratio == 0.0:
        y += padding
    return max(0, min(x, max(0, canvas.width - item.width))), max(0, min(y, max(0, canvas.height - item.height)))


def _rotated(layer: Image.Image, angle: float) -> Image.Image:
    if not angle:
        return layer
    return layer.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)


def _composite(base: Image.Image, layer: Image.Image) -> Image.Image:
    canvas_mode = canvas_mode_for(base, (0, 0, 0, 0))
    if base.mode != "RGBA":
        base = base.convert("RGBA")
    base = base.copy()
    base.alpha_composite(layer)
    if canvas_mode != "RGBA":
        return base.convert(canvas_mode)
    return base


def apply_text_watermark(image: Image.Image, spec: TextWatermark, fonts: FontIndex) -> tuple[Image.Image, str]:
    """Draw ``spec.text`` onto ``image``."""
    if not spec.text or not spec.text.strip():
        raise InvalidArgumentError("watermark text must not be empty")
    opacity = _validate_opacity(spec.opacity)

    shorter = min(image.width, image.height)
    font_size = spec.font_size or max(8, int(shorter * _DEFAULT_FONT_RATIO))
    font, font_name = fonts.resolve(spec.font_family, font_size)

    layer = _render_text_layer(spec, font)
    if spec.tile:
        layer = _tile_layer(Size(*image.size), layer, spec.spacing, spec.rotation)
    else:
        layer = _rotated(layer, spec.rotation)

    if not spec.tile:
        x, y = _gravity_offset(Size(*image.size), Size(*layer.size), spec.position, spec.padding)
        positioned = Image.new("RGBA", image.size, (0, 0, 0, 0))
        positioned.alpha_composite(layer, (x, y))
        layer = positioned

    return _composite(image, _apply_opacity(layer, opacity)), font_name


def _render_text_layer(spec: TextWatermark, font) -> Image.Image:
    """Render just the text (plus shadow) tightly cropped."""
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    try:
        box = measure.textbbox((0, 0), spec.text, font=font, stroke_width=spec.stroke_width)
    except Exception:  # pragma: no cover - exotic fonts
        box = (0, 0, len(spec.text) * 10, 20)
    width = max(1, box[2] - box[0])
    height = max(1, box[3] - box[1])

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    origin = (-box[0], -box[1])

    if spec.shadow:
        shadow_offset = max(1, int(spec.font_size or height) // 20 + 1)
        draw.text(
            (origin[0] + shadow_offset, origin[1] + shadow_offset),
            spec.text,
            font=font,
            fill=(0, 0, 0, 255),
            stroke_width=spec.stroke_width,
            stroke_fill=(0, 0, 0, 255),
        )

    draw.text(
        origin,
        spec.text,
        font=font,
        fill=(*spec.colour, 255),
        stroke_width=spec.stroke_width,
        stroke_fill=(0, 0, 0, 255) if spec.stroke_width else None,
    )
    return layer


def _tile_layer(canvas: Size, tile: Image.Image, spacing: int, rotation: float) -> Image.Image:
    tile = _rotated(tile, rotation)
    step_x = tile.width + max(0, spacing)
    step_y = tile.height + max(0, spacing)
    if step_x < 1 or step_y < 1:  # pragma: no cover - guarded by size math
        raise InvalidArgumentError("watermark tile has no size")

    columns = canvas.width // step_x + 2
    rows = canvas.height // step_y + 2
    if columns * rows > _MAX_TILES:
        raise InvalidArgumentError(
            f"tiling would draw {columns * rows} watermarks, above the {_MAX_TILES} tile limit; "
            "increase spacing or font size"
        )

    layer = Image.new("RGBA", (canvas.width, canvas.height), (0, 0, 0, 0))
    for row in range(rows):
        for column in range(columns):
            # Offset alternate rows so the pattern cannot be trivially removed.
            offset = (step_x // 2) if row % 2 else 0
            x = column * step_x + offset - tile.width // 2
            y = row * step_y - tile.height // 2
            layer.alpha_composite(tile, (x, y))
    return layer


def apply_image_watermark(image: Image.Image, spec: ImageWatermark) -> Image.Image:
    """Composite ``spec.overlay`` onto ``image``."""
    opacity = _validate_opacity(spec.opacity)
    overlay = spec.overlay

    shorter = min(image.width, image.height)
    if spec.scale is not None:
        if spec.scale <= 0:
            raise InvalidArgumentError("watermark scale must be greater than zero")
        target_width = max(1, int(shorter * spec.scale))
    elif overlay.width > image.width or overlay.height > image.height:
        target_width = max(1, int(shorter * _DEFAULT_LOGO_RATIO))
    else:
        target_width = overlay.width

    if target_width != overlay.width:
        ratio = target_width / overlay.width
        overlay = overlay.resize(
            (target_width, max(1, round(overlay.height * ratio))),
            Image.Resampling.LANCZOS,
        )

    if overlay.mode != "RGBA":
        overlay = overlay.convert("RGBA")

    if spec.tile:
        layer = _tile_layer(Size(*image.size), overlay, spec.spacing, 0.0)
    else:
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        x, y = _gravity_offset(Size(*image.size), Size(*overlay.size), spec.position, spec.padding)
        layer.alpha_composite(overlay, (x, y))

    return _composite(image, _apply_opacity(layer, opacity))


__all__ = [
    "ImageWatermark",
    "TextWatermark",
    "apply_image_watermark",
    "apply_text_watermark",
]

"""Composable operation pipelines.

Most real requests are several operations in a row - orient, crop to a ratio,
resize, sharpen, encode - and an agent that has to make five tool calls to do
that pays five round trips and five base64 transfers of the same image. A
pipeline does it in one call and only ever materialises the image once on the
wire.

Operations are pydantic models with a discriminating ``op`` field, so the
generated JSON Schema tells a model exactly which parameters each step accepts.
Validation therefore happens before any pixels are touched.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config import Config
from ..errors import InvalidArgumentError, PictorError
from . import background as background_ops
from . import ops, smartcrop
from .fonts import FontIndex
from .watermark import ImageWatermark, TextWatermark, apply_image_watermark, apply_text_watermark

logger = logging.getLogger(__name__)


class _Operation(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AutoOrientOperation(_Operation):
    op: Literal["auto_orient"] = Field(
        default="auto_orient",
        description="Apply the EXIF orientation tag to the pixels.",
    )


class ResizeOperation(_Operation):
    op: Literal["resize"] = "resize"
    width: int | None = Field(default=None, ge=1, le=100_000, description="Target width in pixels.")
    height: int | None = Field(default=None, ge=1, le=100_000, description="Target height in pixels.")
    percent: float | None = Field(
        default=None, gt=0, le=1000, description="Scale by a percentage; overrides width/height."
    )
    fit: Literal["contain", "cover", "fill", "inside", "outside", "pad"] = Field(
        default="contain",
        description=(
            "contain: fit inside, keep ratio. cover: fill then centre-crop. fill: stretch. "
            "inside: like contain but never enlarge. outside: scale to cover without cropping. "
            "pad: contain then pad to the exact box."
        ),
    )
    filter: str | None = Field(default=None, description="nearest, box, bilinear, hamming, bicubic, lanczos, or auto.")
    only_shrink: bool = Field(default=False, description="Never enlarge the image.")
    background: tuple[int, int, int, int] = Field(
        default=(255, 255, 255, 0), description="Pad/rotate fill colour, RGBA."
    )
    gravity: str = Field(default="center", description="Anchor for cover cropping and padding.")


class CropOperation(_Operation):
    op: Literal["crop"] = "crop"
    box: tuple[int, int, int, int] | None = Field(
        default=None, description="Explicit (left, top, right, bottom) pixel box."
    )
    aspect_ratio: float | None = Field(
        default=None, gt=0, description="Crop to this width/height ratio, e.g. 1.0 or 1.7778."
    )
    gravity: str = Field(default="center", description="Which part of the image to keep when cropping to a ratio.")
    clamp: bool = Field(default=True, description="Clip an out-of-bounds box instead of failing.")
    trim: bool = Field(default=False, description="Auto-remove a uniform border instead of using box/aspect_ratio.")
    trim_tolerance: int = Field(default=12, ge=0, le=255, description="Colour tolerance for border trimming.")

    @field_validator("box")
    @classmethod
    def _check_box(cls, value: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("box must have exactly four values: left, top, right, bottom")
        return value


class RotateOperation(_Operation):
    op: Literal["rotate"] = "rotate"
    angle: float = Field(description="Degrees counter-clockwise.")
    expand: bool = Field(default=True, description="Grow the canvas so nothing is clipped.")
    background: tuple[int, int, int, int] = Field(default=(255, 255, 255, 0))
    filter: str = Field(default="bicubic")


class FlipOperation(_Operation):
    op: Literal["flip"] = "flip"
    direction: Literal["horizontal", "vertical", "both"] = "horizontal"


class SharpenOperation(_Operation):
    op: Literal["sharpen"] = "sharpen"
    amount: float = Field(default=1.0, ge=0, le=10, description="Unsharp-mask strength; 0 disables.")


class BlurOperation(_Operation):
    op: Literal["blur"] = "blur"
    radius: float = Field(gt=0, le=100)


class SmartCropOperation(_Operation):
    op: Literal["smart_crop"] = "smart_crop"
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    method: Literal["attention", "entropy", "center"] = Field(
        default="attention", description="Saliency proxy used to choose the crop window."
    )
    scale: bool = Field(default=True, description="Scale to cover the target before cropping.")


class WatermarkTextOperation(_Operation):
    op: Literal["watermark_text"] = "watermark_text"
    text: str = Field(min_length=1, max_length=500)
    position: str = Field(default="bottom-right")
    opacity: float = Field(default=0.6, ge=0, le=1)
    font_size: int | None = Field(default=None, ge=4, le=4096)
    font_family: str | None = Field(
        default=None, description="Font name from the server's font index, e.g. DejaVuSans-Bold."
    )
    colour: tuple[int, int, int] = Field(default=(255, 255, 255))
    padding: int = Field(default=16, ge=0, le=10_000)
    tile: bool = Field(default=False, description="Repeat the watermark across the whole image.")
    spacing: int = Field(default=0, ge=0, le=10_000)
    rotation: float = Field(default=0.0, ge=-180, le=180)
    shadow: bool = Field(default=True)
    stroke_width: int = Field(default=0, ge=0, le=32)


class WatermarkImageOperation(_Operation):
    op: Literal["watermark_image"] = "watermark_image"
    path: str = Field(description="Path to the watermark image, inside a configured input root.")
    position: str = Field(default="bottom-right")
    opacity: float = Field(default=0.6, ge=0, le=1)
    scale: float | None = Field(
        default=None, gt=0, le=4, description="Width as a fraction of the image's shorter edge."
    )
    padding: int = Field(default=16, ge=0, le=10_000)
    tile: bool = Field(default=False)
    spacing: int = Field(default=0, ge=0, le=10_000)


class BackgroundRemoveOperation(_Operation):
    op: Literal["background_remove"] = "background_remove"
    method: Literal["auto", "color", "ml"] = Field(default="auto")
    tolerance: float = Field(default=32.0, ge=0, le=441)
    softness: float = Field(default=24.0, ge=0, le=441)
    edge_connected: bool = Field(
        default=True, description="Only remove background-coloured regions touching the border."
    )
    model: str = Field(default="u2net", max_length=64)
    background: tuple[int, int, int, int] | None = Field(
        default=None, description="Composite onto this colour instead of leaving transparency."
    )


class FeatherOperation(_Operation):
    op: Literal["feather"] = "feather"
    radius: float = Field(gt=0, le=100)


Operation = Annotated[
    AutoOrientOperation
    | ResizeOperation
    | CropOperation
    | RotateOperation
    | FlipOperation
    | SharpenOperation
    | BlurOperation
    | SmartCropOperation
    | WatermarkTextOperation
    | WatermarkImageOperation
    | BackgroundRemoveOperation
    | FeatherOperation,
    Field(discriminator="op"),
]


class PipelineOutcome:
    """Result of running a pipeline."""

    __slots__ = ("image", "notes", "steps")

    def __init__(self, image: Image.Image, notes: list[str], steps: list[str]) -> None:
        self.image = image
        self.notes = notes
        self.steps = steps


def apply_operations(
    image: Image.Image,
    operations: list[Operation],
    *,
    config: Config,
    fonts: FontIndex,
    overlays: dict[str, Image.Image] | None = None,
    resampler_for=None,
    deadline=None,
) -> PipelineOutcome:
    """Run ``operations`` in order, returning the transformed image.

    Purely synchronous so the caller can run the whole pipeline in a worker
    thread. Any image a step needs from disk (a watermark logo) must be
    pre-resolved by the async layer and handed in through ``overlays``; this
    function never performs I/O of its own, which keeps nested event loops out
    of the picture entirely.

    Each step's result is checked against the configured output-pixel limit
    before the next step runs, so an absurd resize fails immediately rather
    than after allocating.
    """
    notes: list[str] = []
    steps: list[str] = []
    current = image
    limits = config.limits
    available_overlays = overlays or {}

    for index, operation in enumerate(operations):
        if deadline is not None:
            # Checked between steps: a single Pillow call cannot be interrupted,
            # but a long chain of them can be stopped.
            deadline.check(f"pipeline step {index} ({operation.op})")
        try:
            current = _apply_one(
                current,
                operation,
                config=config,
                fonts=fonts,
                overlays=available_overlays,
                resampler_for=resampler_for,
                notes=notes,
            )
        except PictorError as exc:
            # Re-raise with the step context attached, preserving the specific
            # class and machine-readable code. Wrapping everything in
            # InvalidArgumentError would report a resource-limit failure as a
            # bad argument and lose the code a caller branches on.
            raise type(exc)(f"step {index} ({operation.op}): {exc.message}", **exc.details) from exc
        except Exception as exc:
            raise InvalidArgumentError(f"step {index} ({operation.op}) failed: {exc}") from exc

        ops.enforce_output_limits(current.width, current.height, limits)
        steps.append(f"{index}:{operation.op}")

    return PipelineOutcome(image=current, notes=notes, steps=steps)


def watermark_paths(operations: list[Operation]) -> list[str]:
    """Paths that must be pre-resolved before running ``operations``."""
    return [op.path for op in operations if isinstance(op, WatermarkImageOperation)]


def _apply_one(
    image: Image.Image,
    operation: Operation,
    *,
    config: Config,
    fonts: FontIndex,
    overlays: dict[str, Image.Image],
    resampler_for,
    notes: list[str],
) -> Image.Image:
    if isinstance(operation, AutoOrientOperation):
        return ops.auto_orient(image)

    if isinstance(operation, ResizeOperation):
        target = ops.compute_target_size(
            ops.Size(image.width, image.height),
            width=operation.width,
            height=operation.height,
            percent=operation.percent,
            fit=operation.fit,
            only_shrink=operation.only_shrink,
        )
        ops.enforce_output_limits(target.width, target.height, config.limits)
        resampler = resampler_for(image) if resampler_for else None
        return ops.resize(
            image,
            target,
            fit=operation.fit,
            filter_name=operation.filter,
            background=operation.background,
            gravity=operation.gravity,
            resampler=resampler,
            limits=config.limits,
        )

    if isinstance(operation, CropOperation):
        if operation.trim:
            return ops.trim_border(image, tolerance=operation.trim_tolerance)
        if operation.box is not None:
            return ops.crop_box(image, operation.box, clamp=operation.clamp)
        if operation.aspect_ratio is not None:
            return ops.crop_to_aspect(image, operation.aspect_ratio, gravity=operation.gravity)
        raise InvalidArgumentError("crop requires one of box, aspect_ratio or trim")

    if isinstance(operation, RotateOperation):
        return ops.rotate(
            image,
            operation.angle,
            expand=operation.expand,
            background=operation.background,
            filter_name=operation.filter,
        )

    if isinstance(operation, FlipOperation):
        return ops.flip(image, operation.direction)

    if isinstance(operation, SharpenOperation):
        return ops.sharpen(image, amount=operation.amount)

    if isinstance(operation, BlurOperation):
        return ops.blur(image, radius=operation.radius)

    if isinstance(operation, SmartCropOperation):
        target = ops.Size(operation.width, operation.height)
        ops.enforce_output_limits(target.width, target.height, config.limits)
        resampler = resampler_for(image) if resampler_for else None
        if operation.scale:
            return smartcrop.smart_thumbnail(
                image,
                target,
                method=operation.method,
                resampler=resampler,
                limits=config.limits,
            )
        return smartcrop.smart_crop(image, target, method=operation.method)

    if isinstance(operation, WatermarkTextOperation):
        spec = TextWatermark(
            text=operation.text,
            position=operation.position,
            opacity=operation.opacity,
            font_size=operation.font_size,
            font_family=operation.font_family,
            colour=operation.colour,
            padding=operation.padding,
            tile=operation.tile,
            spacing=operation.spacing,
            rotation=operation.rotation,
            shadow=operation.shadow,
            stroke_width=operation.stroke_width,
        )
        result, font_name = apply_text_watermark(image, spec, fonts)
        notes.append(f"text watermark rendered with {font_name}")
        return result

    if isinstance(operation, WatermarkImageOperation):
        overlay = overlays.get(operation.path)
        if overlay is None:
            raise InvalidArgumentError(f"watermark image {operation.path!r} was not pre-loaded; this is a server bug")
        spec = ImageWatermark(
            overlay=overlay,
            position=operation.position,
            opacity=operation.opacity,
            scale=operation.scale,
            padding=operation.padding,
            tile=operation.tile,
            spacing=operation.spacing,
        )
        return apply_image_watermark(image, spec)

    if isinstance(operation, BackgroundRemoveOperation):
        result, background_notes = background_ops.remove_background(
            image,
            method=operation.method,
            tolerance=operation.tolerance,
            softness=operation.softness,
            edge_connected=operation.edge_connected,
            model=operation.model,
            background=operation.background,
        )
        notes.extend(background_notes)
        return result

    if isinstance(operation, FeatherOperation):
        return background_ops.feather_alpha(image, operation.radius)

    raise InvalidArgumentError(f"unsupported operation {getattr(operation, 'op', operation)!r}")


_OPERATION_MODELS: tuple[tuple[str, type[_Operation]], ...] = (
    ("auto_orient", AutoOrientOperation),
    ("resize", ResizeOperation),
    ("crop", CropOperation),
    ("rotate", RotateOperation),
    ("flip", FlipOperation),
    ("sharpen", SharpenOperation),
    ("blur", BlurOperation),
    ("smart_crop", SmartCropOperation),
    ("watermark_text", WatermarkTextOperation),
    ("watermark_image", WatermarkImageOperation),
    ("background_remove", BackgroundRemoveOperation),
    ("feather", FeatherOperation),
)

#: Operation names accepted by the pipeline tools.
OPERATION_NAMES: list[str] = [name for name, _ in _OPERATION_MODELS]


def describe_operations() -> list[dict[str, Any]]:
    """Operation catalogue for the capabilities tool."""
    return [
        {
            "op": name,
            "parameters": sorted(model.model_fields),
        }
        for name, model in _OPERATION_MODELS
    ]


__all__ = [
    "OPERATION_NAMES",
    "Operation",
    "PipelineOutcome",
    "apply_operations",
    "describe_operations",
    "watermark_paths",
]

"""Analysis and web-delivery tools: compare, and responsive optimization.

``image_compare`` closes the loop on an edit: an agent that resized or
recompressed an image can verify the result rather than assume it. SSIM is the
headline number because it tracks perceived change, while RMSE and the
perceptual-hash distance catch the cases SSIM is blind to.

``image_optimize_web`` encodes one source into several widths with per-variant
byte ceilings, which is what "make this web-ready" actually means: a set of
files, not one file.
"""

from __future__ import annotations

import base64
import logging
from typing import Annotated, Any

import anyio
from mcp.server.mcpserver import MCPServer
from PIL import Image, ImageFilter
from pydantic import Field

from ..errors import InvalidArgumentError
from ..imaging.compare import compare_images, difference_image
from ..imaging.encode import EncodedImage, encode
from ..imaging.formats import FormatSpec, resolve_output_format
from ..imaging.loader import LoadedImage
from ..imaging.ops import Size, compute_target_size, scale_exact
from ..models import ImageResult
from ..outputs import build_result, default_filename, human_bytes, size_change_for
from ..security.limits import check_output_geometry
from .basic import _collect_frames, _search_quality, _summarise_input
from .context import SUBDIR_EDIT, SUBDIR_WEB, ToolContext, encode_options, tool_guard

logger = logging.getLogger(__name__)

#: Width used for the low-quality image placeholder.
_PLACEHOLDER_WIDTH = 32

#: Variants at or below this width are placeholders, not srcset entries.
_PLACEHOLDER_THRESHOLD = 64


def register(server: MCPServer, ctx: ToolContext) -> None:
    """Register analysis tools."""

    # ----------------------------------------------------------------- compare
    @server.tool(
        name="image_compare",
        title="Compare two images",
        description=(
            "Measure how different two images are: SSIM, RMSE, PSNR, mean and max pixel difference, "
            "the fraction of changed pixels, and perceptual-hash distance. Use it to verify a "
            "transform or to spot duplicate images."
        ),
    )
    @tool_guard
    async def image_compare(
        path: Annotated[str | None, Field(description="Path to the first image.")] = None,
        base64_data: Annotated[str | None, Field(description="Inline base64 for the first image.")] = None,
        url: Annotated[str | None, Field(description="URL for the first image.")] = None,
        compare_to_path: Annotated[str | None, Field(description="Path to the second image.")] = None,
        compare_to_base64: Annotated[str | None, Field(description="Inline base64 for the second image.")] = None,
        compare_to_url: Annotated[str | None, Field(description="URL for the second image.")] = None,
        align: Annotated[
            bool,
            Field(description="Resize the second image to the first's dimensions before comparing."),
        ] = True,
        change_threshold: Annotated[
            int,
            Field(ge=0, le=255, description="Per-pixel difference counted as a change."),
        ] = 8,
        create_diff_image: Annotated[
            bool,
            Field(description="Also write a visual difference image (white where identical)."),
        ] = False,
        diff_amplify: Annotated[float, Field(ge=1, le=20, description="Amplify the diff visually.")] = 4.0,
    ) -> ImageResult:
        with ctx.gate.slot():
            first = await ctx.load_input(path=path, base64_data=base64_data, url=url)
            try:
                second = await ctx.load_input(
                    path=compare_to_path,
                    base64_data=compare_to_base64,
                    url=compare_to_url,
                )
                try:
                    metrics = await anyio.to_thread.run_sync(
                        lambda: compare_images(
                            first.image,
                            second.image,
                            align=align,
                            change_threshold=change_threshold,
                        ).to_public_dict()
                    )
                    outputs = []
                    notes: list[str] = []

                    if create_diff_image:
                        diff = await anyio.to_thread.run_sync(
                            lambda: difference_image(first.image, second.image, amplify=diff_amplify)
                        )
                        spec = resolve_output_format("png")
                        encoded = await anyio.to_thread.run_sync(
                            lambda: encode(diff, spec, encode_options(lossless=True))
                        )
                        outputs.append(
                            await ctx.store_encoded(encoded, subdirectory=SUBDIR_EDIT, filename="difference.png")
                        )
                        notes.append("the difference image is white where the two images match")

                    result = ImageResult(
                        operation="image_compare",
                        outputs=[item.file for item in outputs],
                        input={"a": _summarise_input(first), "b": _summarise_input(second)},
                        metrics=metrics,
                        notes=notes,
                    )
                    text = "\n".join(
                        [
                            f"Comparison: {_verdict(metrics)}",
                            f"SSIM: {metrics['ssim']} (1.0 is identical)",
                            f"RMSE: {metrics['rmse']}  PSNR: {metrics['psnrDb']} dB",
                            f"Changed pixels: {metrics['changedPixelRatio'] * 100:.3f}% (threshold {change_threshold})",
                            f"Perceptual hash distance: {metrics['perceptualHashDistance']}/64",
                            *([f"Diff image: {outputs[0].file.path}"] if outputs else []),
                        ]
                    )
                    return build_result(result, text)
                finally:
                    second.close()
            finally:
                first.close()

    # ----------------------------------------------------------- optimize web
    @server.tool(
        name="image_optimize_web",
        title="Produce responsive web variants",
        description=(
            "Turn one source image into a set of web-ready variants at several widths, each "
            "optionally constrained to a maximum file size, plus a tiny blurred placeholder and a "
            "ready-to-paste srcset. This is the one call to use for 'make this web-ready'."
        ),
    )
    @tool_guard
    async def image_optimize_web(
        path: Annotated[str | None, Field(description="Path to the source image.")] = None,
        base64_data: Annotated[str | None, Field(description="Inline base64 source image.")] = None,
        url: Annotated[str | None, Field(description="http(s) URL to fetch.")] = None,
        widths: Annotated[
            list[Annotated[int, Field(ge=1, le=100_000)]] | None,
            Field(
                max_length=12,
                description="Target widths, e.g. [480, 960, 1440]. Defaults to a sensible ladder.",
            ),
        ] = None,
        output_format: Annotated[str, Field(description="Output codec: webp, avif or jpeg.")] = "webp",
        quality: Annotated[int, Field(ge=1, le=100, description="Starting quality for each variant.")] = 82,
        max_bytes_per_variant: Annotated[
            int | None,
            Field(gt=0, description="Per-variant byte ceiling; quality is searched down per width to fit."),
        ] = None,
        create_placeholder: Annotated[
            bool,
            Field(description="Also produce a tiny blurred placeholder and its base64 data URI."),
        ] = True,
        strip_metadata: Annotated[bool | None, Field(description="Strip metadata from every variant.")] = True,
    ) -> ImageResult:
        with ctx.gate.slot():
            loaded = await ctx.load_input(path=path, base64_data=base64_data, url=url)
            try:
                spec = resolve_output_format(output_format)
                if spec.key not in {"webp", "avif", "jpeg", "png"}:
                    raise InvalidArgumentError(
                        "optimize_web targets web formats: use webp, avif, jpeg or png",
                        format=spec.key,
                    )
                ladder = _width_ladder(widths, loaded.geometry.width)
                # One budget for the whole call, shared by every variant and
                # every quality search within it.
                deadline = ctx.deadline()
                notes: list[str] = list(loaded.notes)
                produced: list[tuple[EncodedImage, str]] = []
                variant_metrics: list[dict[str, Any]] = []

                animated = ctx.wants_animation(loaded, spec)

                for target_width in ladder:
                    target = compute_target_size(
                        Size(loaded.geometry.width, loaded.geometry.height),
                        width=target_width,
                        fit="contain",
                    )
                    encoded, variant_notes, used_quality = await _encode_variant(
                        ctx,
                        loaded,
                        spec,
                        target=target,
                        quality=quality,
                        max_bytes=max_bytes_per_variant,
                        strip_metadata=strip_metadata,
                        animated=animated,
                        deadline=deadline,
                    )
                    notes.extend(variant_notes)
                    produced.append((encoded, default_filename(loaded, spec, suffix=f"{encoded.width}w")))
                    variant_metrics.append(
                        {
                            "width": encoded.width,
                            "height": encoded.height,
                            "bytes": encoded.byte_size,
                            "quality": used_quality,
                        }
                    )

                stored = await ctx.store_many(produced, subdirectory=SUBDIR_WEB)
                outputs = [item.file for item in stored]

                metrics: dict[str, Any] = {
                    "variants": variant_metrics,
                    "animated": animated,
                }

                if create_placeholder:
                    placeholder_file, data_uri = await _placeholder(ctx, loaded, spec)
                    if placeholder_file is not None:
                        outputs.append(placeholder_file.file)
                        metrics["placeholderDataUri"] = data_uri
                        metrics["placeholderBytes"] = placeholder_file.file.byte_size

                srcset = ", ".join(
                    f"{item.path} {item.width}w" for item in outputs if item.width > _PLACEHOLDER_THRESHOLD
                )
                metrics["srcset"] = srcset
                metrics["totalBytes"] = sum(item.byte_size for item in outputs)

                result = ImageResult(
                    operation="image_optimize_web",
                    outputs=outputs,
                    input=_summarise_input(loaded),
                    sizeChange=size_change_for(loaded.byte_size, metrics["totalBytes"]),
                    notes=notes,
                    metrics=metrics,
                )
                text = "\n".join(
                    [
                        f"Produced {len(outputs)} web variants from a {loaded.geometry.width}px source.",
                        *[
                            f"  {item.width}x{item.height} {item.format} {human_bytes(item.byte_size)} -> {item.path}"
                            for item in outputs
                            if item.width > _PLACEHOLDER_THRESHOLD
                        ],
                        *(
                            [f"  placeholder {human_bytes(metrics.get('placeholderBytes'))} (data URI in metrics)"]
                            if "placeholderDataUri" in metrics
                            else []
                        ),
                        f"srcset: {srcset}" if srcset else "",
                    ]
                ).strip()
                return build_result(result, text)
            finally:
                loaded.close()


# --------------------------------------------------------------------- helpers


def _verdict(metrics: dict[str, Any]) -> str:
    """Turn the numbers into a plain-language judgement."""
    if metrics["identical"]:
        return "the images are pixel-identical"
    ssim = metrics["ssim"]
    distance = metrics["perceptualHashDistance"]
    if ssim >= 0.995 and distance <= 4:
        return "visually identical (differences are imperceptible)"
    if ssim >= 0.98:
        return "visually near-identical"
    if ssim >= 0.90:
        return "similar, with visible but minor differences"
    if ssim >= 0.70:
        return "noticeably different"
    return "substantially different — likely different images"


#: Hard ceiling on a requested variant width, independent of the source size.
#: The runtime geometry check is authoritative; this just fails earlier with a
#: message that names the offending value.
_MAX_REQUESTED_WIDTH = 100_000


def _width_ladder(requested: list[int] | None, source_width: int) -> list[int]:
    """Pick target widths, never exceeding the source by default."""
    if requested:
        cleaned = sorted({int(value) for value in requested if value > 0})
        if not cleaned:
            raise InvalidArgumentError("widths must contain at least one positive integer")
        for width in cleaned:
            if width > _MAX_REQUESTED_WIDTH:
                raise InvalidArgumentError(
                    f"requested width {width} exceeds the {_MAX_REQUESTED_WIDTH}px limit",
                    width=width,
                    limit=_MAX_REQUESTED_WIDTH,
                )
        return cleaned
    ladder = [width for width in (480, 960, 1440, 1920) if width <= source_width]
    if not ladder:
        return [max(1, source_width)]
    if ladder[-1] < source_width and source_width <= 2560:
        ladder.append(source_width)
    return ladder


async def _encode_variant(
    ctx: ToolContext,
    loaded: LoadedImage,
    spec: FormatSpec,
    *,
    target: Size,
    quality: int,
    max_bytes: int | None,
    strip_metadata: bool | None,
    animated: bool,
    deadline=None,
) -> tuple[EncodedImage, list[str], int | None]:
    """Scale to ``target`` and encode, sizing the search against the scaled copy.

    The quality search runs against the *resized* bitmap. Searching against the
    full-size image and then encoding a smaller one would pick a quality that
    has nothing to do with the bytes actually written.
    """
    notes: list[str] = []

    # This path encodes directly instead of going through the pipeline, so it
    # has to enforce the output geometry itself. Without this a single absurd
    # width (say 100000000) reaches Pillow and dies with a MemoryError, or
    # allocates gigabytes on the way there.
    check_output_geometry(target.width, target.height, ctx.config.limits)

    if animated:
        if max_bytes is not None:
            notes.append(
                "per-variant size targeting is skipped for animated output; "
                "quality is applied as requested to every frame"
            )
        frames, durations = await anyio.to_thread.run_sync(_collect_frames, loaded)
        resized = await anyio.to_thread.run_sync(
            lambda: [frame.resize((target.width, target.height), Image.Resampling.LANCZOS) for frame in frames]
        )
        options = encode_options(quality=quality, strip_metadata=strip_metadata)
        encoded = await ctx.encode_frames(resized, spec, options, durations=durations, loop=0)
        return encoded, notes, quality

    def scale():
        return scale_exact(loaded.image, target, resampler=ctx.registry.resampler_for(loaded.image))

    scaled = await anyio.to_thread.run_sync(scale)

    options = encode_options(quality=quality, strip_metadata=strip_metadata)
    used_quality: int | None = quality
    if max_bytes is not None:
        options, search_notes = await _search_quality(
            ctx,
            scaled,
            spec,
            quality=quality,
            target_bytes=max_bytes,
            strip_metadata=strip_metadata,
            effort=None,
            lossless=False,
            deadline=deadline,
        )
        notes.extend(search_notes)
        used_quality = options.quality

    encoded = await anyio.to_thread.run_sync(encode, scaled, spec, options)
    return encoded, notes, used_quality


async def _placeholder(
    ctx: ToolContext,
    loaded: LoadedImage,
    spec: FormatSpec,
) -> tuple[Any | None, str | None]:
    """Build a tiny blurred placeholder and its data URI."""
    placeholder_spec = resolve_output_format("jpeg" if spec.key == "webp" else "webp")

    def work() -> EncodedImage:
        ratio = _PLACEHOLDER_WIDTH / loaded.image.width
        tiny = loaded.image.resize(
            (_PLACEHOLDER_WIDTH, max(1, int(loaded.image.height * ratio))),
            Image.Resampling.LANCZOS,
        ).convert("RGB")
        tiny = tiny.filter(ImageFilter.GaussianBlur(1.2))
        return encode(tiny, placeholder_spec, encode_options(quality=35, strip_metadata=True))

    encoded = await anyio.to_thread.run_sync(work)
    filename = default_filename(loaded, placeholder_spec, suffix="placeholder")
    stored = await ctx.store_encoded(encoded, subdirectory=SUBDIR_WEB, filename=filename)
    data_uri = f"data:{placeholder_spec.mime};base64,{base64.b64encode(encoded.data).decode()}"
    return stored, data_uri


__all__ = ["register"]

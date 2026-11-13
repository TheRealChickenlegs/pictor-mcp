"""Core single-image tools: inspect, convert, resize, compress, crop, rotate, thumbnail.

Each tool takes exactly one input (``path``, ``base64_data`` or ``url``) and
returns the standard result envelope. Shared parameter documentation lives in
the ``Annotated`` descriptions so every tool's generated JSON Schema explains
itself to a model without a separate prompt.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..errors import InvalidArgumentError
from ..imaging.encode import EncodeOptions
from ..imaging.formats import OUTPUT_FORMATS, resolve_output_format
from ..imaging.loader import LoadedImage
from ..imaging.pipeline import CropOperation, FlipOperation, ResizeOperation, RotateOperation, SharpenOperation
from ..models import BackendReport, ImageResult, InfoResult
from ..outputs import human_bytes
from .context import (
    SUBDIR_COMPRESS,
    SUBDIR_CONVERT,
    SUBDIR_EDIT,
    SUBDIR_RESIZE,
    ToolContext,
    encode_options,
    tool_guard,
)

logger = logging.getLogger(__name__)

_PathArg = Annotated[
    str | None,
    Field(description="Image path inside an allowed input root."),
]
_Base64Arg = Annotated[
    str | None,
    Field(description="Base64 image data; data URIs are accepted."),
]
_UrlArg = Annotated[
    str | None,
    Field(description="HTTP(S) image URL; fetching may be disabled."),
]
_FormatArg = Annotated[
    str | None,
    Field(description="Output format; defaults to the input format."),
]
_QualityArg = Annotated[
    int | None,
    Field(ge=1, le=100, description="Lossy quality (1-100)."),
]
_ReturnImageArg = Annotated[
    bool,
    Field(description="Include the image inline when server configuration permits."),
]
_ReturnBase64Arg = Annotated[
    bool,
    Field(description="Include encoded image bytes in the result."),
]
_OutputNameArg = Annotated[
    str | None,
    Field(description="Output filename override."),
]
_StripArg = Annotated[
    bool | None,
    Field(description="Strip EXIF/GPS/ICC metadata; defaults to the server setting."),
]


def _output_spec_key(loaded: LoadedImage, requested: str | None) -> tuple[str, list[str]]:
    """Pick the output codec, falling back when the input is read-only."""
    notes: list[str] = []
    if requested:
        return requested, notes
    if loaded.spec.key in OUTPUT_FORMATS:
        return loaded.spec.key, notes
    notes.append(f"{loaded.spec.label} cannot be written; defaulted the output format to PNG")
    return "png", notes


def _summarise_input(loaded: LoadedImage) -> dict[str, Any]:
    return {
        "width": loaded.geometry.width,
        "height": loaded.geometry.height,
        "format": loaded.original_format,
        "mode": loaded.image.mode,
        "byteSize": loaded.byte_size,
        "frames": loaded.frames,
        "hasAlpha": loaded.has_alpha,
    }


def _input_label(loaded: LoadedImage) -> str:
    if loaded.source.kind == "path":
        from pathlib import Path

        return Path(loaded.source.value).name
    return "inline image" if loaded.source.kind == "base64" else "remote image"


def register(server: MCPServer, ctx: ToolContext) -> None:
    """Register the core tools on ``server``."""

    # ------------------------------------------------------------ capabilities
    @server.tool(
        name="image_capabilities",
        title="Image server capabilities",
        description="Report supported formats and operations, limits, security settings, and active acceleration.",
    )
    @tool_guard(ctx)
    async def image_capabilities() -> BackendReport:
        from ..imaging.background import ml_available
        from ..imaging.formats import public_format_catalogue
        from ..imaging.pipeline import describe_operations
        from ..imaging.vector import renderer_available, renderer_label
        from ..outputs import build_result

        ml_ok, ml_detail = ml_available()
        security = {
            "inputRoots": [str(p) for p in ctx.config.input_roots],
            "outputRoot": str(ctx.config.output_root),
            "pathConfinement": "reads and writes are confined to the roots above; symlink escapes and traversal are rejected",
            "networkFetch": {
                "enabled": ctx.config.fetch.enabled,
                "allowedHosts": list(ctx.config.fetch.allowed_hosts) or "any public host",
                "blockedRanges": "private, loopback, link-local, CGNAT, reserved and multicast addresses",
                "maxBytes": ctx.config.fetch.max_bytes,
            },
            "metadataStrippedByDefault": ctx.config.strip_metadata,
            "httpAuthEnabled": bool(ctx.config.http.auth_token),
            "outputUrlsEnabled": ctx.config.http.serve_outputs,
            "dnsRebindingProtection": ctx.config.http.dns_rebinding_protection,
            "backgroundRemovalMl": {"available": ml_ok, "detail": ml_detail},
            # SVG is the one input rendered outside this process, so the isolation
            # boundary is reported rather than left implicit: a caller that cares
            # whether a vector file is accepted needs to know the renderer exists.
            "svg": {
                "enabled": ctx.config.svg.enabled,
                "renderer": renderer_label(),
                "rendererAvailable": renderer_available(),
                "isolation": "rendered in a separate short-lived process, never in the server itself",
                "externalReferences": "file:, http: and https: references are never resolved",
                "systemFonts": ctx.config.svg.system_fonts,
                "maxDepth": ctx.config.svg.max_depth,
                "defaultSize": ctx.config.svg.default_size,
                "timeoutSeconds": ctx.config.svg.timeout_seconds,
            },
        }
        limits = {
            "maxFileBytes": ctx.config.limits.max_file_bytes,
            "maxPixels": ctx.config.limits.max_pixels,
            "maxDimension": ctx.config.limits.max_dimension,
            "maxFrames": ctx.config.limits.max_frames,
            "maxOutputPixels": ctx.config.limits.max_output_pixels,
            "maxBatchFiles": ctx.config.limits.max_batch_files,
            "opTimeoutSeconds": ctx.config.limits.op_timeout_seconds,
            "maxConcurrency": ctx.config.limits.max_concurrency,
            "maxInlineImageBytes": ctx.config.limits.max_inline_image_bytes,
        }
        report = BackendReport(
            server={
                "name": "pictor-mcp",
                "version": ctx.server_version,
                "transport": ctx.config.transport,
            },
            protocol={
                "supported": ["2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26"],
                "negotiation": "automatic: modern stateless requests and legacy initialize-handshake clients are both served",
                "structuredContent": True,
                "inlineImages": ctx.config.inline_images,
                "resourceLinks": True,
            },
            formats=public_format_catalogue(svg_enabled=ctx.config.svg.enabled),
            operations=describe_operations(),
            limits=limits,
            security=security,
            backends=ctx.registry.to_public_dict(),
            gpu=ctx.registry.gpu_device_info(),
            notes=[
                "Image is written to the output root and returned as a URL in the result text.",
                "Inline image bytes are off unless PICTOR_INLINE_IMAGES=true; prefer the markdown URL.",
                "Use image_transform to chain several steps in one call.",
                "Call image_list_inputs to discover readable files when a path is not known.",
            ],
        )
        text_lines = [
            "pictor-mcp image server is ready.",
            f"Transport: {ctx.config.transport}.",
            f"Acceleration: {ctx.registry.active}.",
            f"Formats: {', '.join(f['format'] for f in report.formats['writable'])}.",
            f"Input roots: {', '.join(str(p) for p in ctx.config.input_roots)}.",
            f"Network fetch: {'enabled' if ctx.config.fetch.enabled else 'disabled'}.",
            f"ML background removal: {'available' if ml_ok else 'unavailable — ' + ml_detail}.",
        ]
        return build_result(report, "\n".join(text_lines))

    # -------------------------------------------------------------------- info
    @server.tool(
        name="image_info",
        title="Inspect an image",
        description="Inspect image dimensions, format, animation, EXIF, perceptual hashes, and dominant colours.",
    )
    @tool_guard(ctx)
    async def image_info(
        path: _PathArg = None,
        base64_data: _Base64Arg = None,
        url: _UrlArg = None,
        include_exif: Annotated[bool, Field(description="Include decoded EXIF tags.")] = True,
        include_hashes: Annotated[bool, Field(description="Include perceptual hashes for duplicate detection.")] = True,
        include_colours: Annotated[bool, Field(description="Include the dominant colour palette.")] = False,
        max_exif_entries: Annotated[int, Field(ge=1, le=500, description="Cap on returned EXIF entries.")] = 60,
    ) -> InfoResult:
        import anyio

        from ..outputs import build_result

        with ctx.gate.slot():
            loaded = await ctx.load_input(path=path, base64_data=base64_data, url=url)
            try:
                exif = _extract_exif(loaded, max_entries=max_exif_entries) if include_exif else None
                hashes = None
                colours = None

                def work() -> tuple[dict[str, str], list[dict[str, Any]] | None]:
                    from ..imaging.compare import perceptual_hashes

                    computed = perceptual_hashes(loaded.image)
                    palette = _dominant_colours(loaded) if include_colours else None
                    return computed, palette

                if include_hashes or include_colours:
                    computed, colours = await anyio.to_thread.run_sync(work)
                    if include_hashes:
                        hashes = computed

                info = {
                    **_summarise_input(loaded),
                    "mimeType": loaded.mime_type,
                    "isAnimated": loaded.is_animated,
                    "source": loaded.source.redacted(),
                    "origin": loaded.image.info.get("pictor_origin"),
                }
                result = InfoResult(
                    image=info,
                    exif=exif,
                    hashes=hashes,
                    dominantColours=colours,
                    notes=list(loaded.notes),
                )
                text = [
                    f"{info['width']}x{info['height']} {str(info['format']).upper()}, "
                    f"{info['mode']}, {human_bytes(info['byteSize'])}",
                ]
                if exif:
                    text.append(
                        f"EXIF: {len(exif)} tags" + (f" (camera: {exif.get('Model')})" if exif.get("Model") else "")
                    )
                if hashes:
                    text.append(f"pHash: {hashes['phash']}")
                if colours:
                    text.append("Dominant colours: " + ", ".join(c["hex"] for c in colours))
                return build_result(result, "\n".join(text))
            finally:
                loaded.close()

    # ----------------------------------------------------------------- convert
    @server.tool(
        name="image_convert",
        title="Convert image format",
        description="Convert image formats; SVG is accepted as input but not as an output format.",
    )
    @tool_guard(ctx)
    async def image_convert(
        path: _PathArg = None,
        base64_data: _Base64Arg = None,
        url: _UrlArg = None,
        target_format: Annotated[str, Field(description="Target output format.")] = "webp",
        quality: _QualityArg = None,
        lossless: Annotated[bool, Field(description="Prefer lossless encoding when supported.")] = False,
        progressive: Annotated[bool, Field(description="Write a progressive JPEG.")] = False,
        effort: Annotated[
            int | None,
            Field(
                ge=0,
                le=10,
                description="Encoder effort; higher is slower and usually smaller.",
            ),
        ] = None,
        background: Annotated[
            tuple[int, int, int] | None,
            Field(description="RGB background for formats without alpha."),
        ] = None,
        strip_metadata: _StripArg = None,
        keep_icc: Annotated[bool, Field(description="Preserve the ICC profile.")] = False,
        output_name: _OutputNameArg = None,
        return_image: _ReturnImageArg = True,
        return_base64: _ReturnBase64Arg = False,
    ) -> ImageResult:
        with ctx.gate.slot():
            loaded = await ctx.load_input(path=path, base64_data=base64_data, url=url)
            try:
                fmt, notes = _output_spec_key(loaded, target_format)
                options = encode_options(
                    quality=quality,
                    lossless=lossless,
                    progressive=progressive,
                    effort=effort,
                    background=background,
                    strip_metadata=strip_metadata,
                    keep_icc=keep_icc,
                )
                spec = resolve_output_format(fmt)
                encoded = await _encode_loaded(ctx, loaded, spec, options)
                filename = ctx.output_filename(output_name, spec, loaded)
                stored = await ctx.store_encoded(
                    encoded, subdirectory=SUBDIR_CONVERT, filename=filename, include_base64=return_base64
                )
                return ctx.builder.build_image_result(
                    operation="image_convert",
                    outputs=[stored],
                    input_summary=_summarise_input(loaded),
                    input_bytes=loaded.byte_size,
                    notes=notes + encoded.notes,
                    inline=return_image,
                )
            finally:
                loaded.close()

    # ------------------------------------------------------------------ resize
    @server.tool(
        name="image_resize",
        title="Resize an image",
        description="Resize by width, height, percentage, or box; supports fit modes and format conversion.",
    )
    @tool_guard(ctx)
    async def image_resize(
        path: _PathArg = None,
        base64_data: _Base64Arg = None,
        url: _UrlArg = None,
        width: Annotated[int | None, Field(ge=1, le=100_000, description="Width in pixels.")] = None,
        height: Annotated[int | None, Field(ge=1, le=100_000, description="Height in pixels.")] = None,
        percent: Annotated[
            float | None, Field(gt=0, le=1000, description="Scale by this percentage of the original.")
        ] = None,
        fit: Annotated[
            str,
            Field(description="contain, cover, fill, inside, outside, or pad; see operation semantics."),
        ] = "contain",
        filter: Annotated[
            str | None,
            Field(description="Resampling filter: nearest, box, bilinear, hamming, bicubic, lanczos, or auto."),
        ] = None,
        only_shrink: Annotated[bool, Field(description="Never enlarge; leave smaller images untouched.")] = False,
        background: Annotated[
            tuple[int, int, int, int] | None,
            Field(description="RGBA fill for padding."),
        ] = None,
        gravity: Annotated[str, Field(description="Anchor for cropping or padding.")] = "center",
        output_format: _FormatArg = None,
        quality: _QualityArg = None,
        strip_metadata: _StripArg = None,
        output_name: _OutputNameArg = None,
        return_image: _ReturnImageArg = True,
        return_base64: _ReturnBase64Arg = False,
    ) -> ImageResult:
        with ctx.gate.slot():
            loaded = await ctx.load_input(path=path, base64_data=base64_data, url=url)
            try:
                fmt, notes = _output_spec_key(loaded, output_format)
                operation = ResizeOperation(
                    width=width,
                    height=height,
                    percent=percent,
                    fit=fit,  # type: ignore[arg-type]
                    filter=filter,
                    only_shrink=only_shrink,
                    background=background or (255, 255, 255, 0),
                    gravity=gravity,
                )
                options = encode_options(quality=quality, strip_metadata=strip_metadata)
                outputs, pipe_notes, steps, encoded = await ctx.process(
                    loaded.image,
                    [operation],
                    fmt=fmt,
                    options=options,
                    subdirectory=SUBDIR_RESIZE,
                    filename_suffix=_dimension_suffix(width, height, percent),
                    source=loaded,
                    include_base64=return_base64,
                    output_name=output_name,
                )
                return ctx.builder.build_image_result(
                    operation="image_resize",
                    outputs=outputs,
                    input_summary=_summarise_input(loaded),
                    input_bytes=loaded.byte_size,
                    notes=notes + pipe_notes + encoded.notes + loaded.notes,
                    steps=steps,
                    inline=return_image,
                )
            finally:
                loaded.close()

    # --------------------------------------------------------------- compress
    @server.tool(
        name="image_compress",
        title="Compress an image",
        description="Compress by quality or target file size; reports the achieved saving.",
    )
    @tool_guard(ctx)
    async def image_compress(
        path: _PathArg = None,
        base64_data: _Base64Arg = None,
        url: _UrlArg = None,
        quality: _QualityArg = None,
        target_size_kb: Annotated[
            float | None,
            Field(gt=0, le=1_048_576, description="Maximum output size in KB; quality is adjusted to fit."),
        ] = None,
        target_format: _FormatArg = None,
        lossless: Annotated[bool, Field(description="Prefer lossless compression where supported.")] = False,
        effort: Annotated[
            int | None, Field(ge=0, le=10, description="Encoder effort; higher is slower and smaller.")
        ] = None,
        strip_metadata: _StripArg = None,
        output_name: _OutputNameArg = None,
        return_image: _ReturnImageArg = True,
        return_base64: _ReturnBase64Arg = False,
    ) -> ImageResult:
        with ctx.gate.slot():
            loaded = await ctx.load_input(path=path, base64_data=base64_data, url=url)
            try:
                fmt, notes = _output_spec_key(loaded, target_format)
                spec = resolve_output_format(fmt)
                if target_size_kb is not None and lossless:
                    raise InvalidArgumentError(
                        "target_size_kb cannot be combined with lossless encoding: a lossless codec "
                        "has no quality dial to trade against size"
                    )

                options, search_notes = await _search_quality(
                    ctx,
                    loaded.image,
                    spec,
                    quality=quality,
                    target_bytes=int(target_size_kb * 1024) if target_size_kb else None,
                    strip_metadata=strip_metadata,
                    effort=effort,
                    lossless=lossless,
                    deadline=ctx.deadline(),
                )
                encoded = await _encode_loaded(ctx, loaded, spec, options)
                filename = ctx.output_filename(output_name, spec, loaded)
                stored = await ctx.store_encoded(
                    encoded, subdirectory=SUBDIR_COMPRESS, filename=filename, include_base64=return_base64
                )
                metrics = {
                    "qualityUsed": options.quality,
                    "lossless": options.lossless,
                    "inputBytes": loaded.byte_size,
                    "outputBytes": encoded.byte_size,
                }
                return ctx.builder.build_image_result(
                    operation="image_compress",
                    outputs=[stored],
                    input_summary=_summarise_input(loaded),
                    input_bytes=loaded.byte_size,
                    notes=notes + search_notes + encoded.notes + loaded.notes,
                    metrics=metrics,
                    inline=return_image,
                )
            finally:
                loaded.close()

    # ------------------------------------------------------------------- crop
    @server.tool(
        name="image_crop",
        title="Crop an image",
        description="Crop by pixel box, aspect ratio, or uniform-border trim.",
    )
    @tool_guard(ctx)
    async def image_crop(
        path: _PathArg = None,
        base64_data: _Base64Arg = None,
        url: _UrlArg = None,
        box: Annotated[
            tuple[int, int, int, int] | None,
            Field(description="Crop box: left, top, right, bottom."),
        ] = None,
        aspect_ratio: Annotated[
            float | None,
            Field(gt=0, le=100, description="Crop width ÷ height ratio."),
        ] = None,
        gravity: Annotated[str, Field(description="Crop anchor, e.g. center or top-left.")] = "center",
        trim: Annotated[bool, Field(description="Trim a uniform border.")] = False,
        trim_tolerance: Annotated[int, Field(ge=0, le=255, description="Border colour tolerance.")] = 12,
        clamp: Annotated[bool, Field(description="Clamp crop box to image bounds.")] = True,
        output_format: _FormatArg = None,
        quality: _QualityArg = None,
        strip_metadata: _StripArg = None,
        output_name: _OutputNameArg = None,
        return_image: _ReturnImageArg = True,
        return_base64: _ReturnBase64Arg = False,
    ) -> ImageResult:
        with ctx.gate.slot():
            loaded = await ctx.load_input(path=path, base64_data=base64_data, url=url)
            try:
                if not (trim or box is not None or aspect_ratio is not None):
                    raise InvalidArgumentError("provide one of box, aspect_ratio or trim=true")
                fmt, notes = _output_spec_key(loaded, output_format)
                operation = CropOperation(
                    box=box,
                    aspect_ratio=aspect_ratio,
                    gravity=gravity,
                    clamp=clamp,
                    trim=trim,
                    trim_tolerance=trim_tolerance,
                )
                options = encode_options(quality=quality, strip_metadata=strip_metadata)
                outputs, pipe_notes, steps, encoded = await ctx.process(
                    loaded.image,
                    [operation],
                    fmt=fmt,
                    options=options,
                    subdirectory=SUBDIR_EDIT,
                    filename_suffix="crop",
                    source=loaded,
                    include_base64=return_base64,
                    output_name=output_name,
                )
                return ctx.builder.build_image_result(
                    operation="image_crop",
                    outputs=outputs,
                    input_summary=_summarise_input(loaded),
                    input_bytes=loaded.byte_size,
                    notes=notes + pipe_notes + encoded.notes + loaded.notes,
                    steps=steps,
                    inline=return_image,
                )
            finally:
                loaded.close()

    # ----------------------------------------------------------------- rotate
    @server.tool(
        name="image_rotate",
        title="Rotate or flip an image",
        description="Rotate, flip, or apply EXIF orientation.",
    )
    @tool_guard(ctx)
    async def image_rotate(
        path: _PathArg = None,
        base64_data: _Base64Arg = None,
        url: _UrlArg = None,
        angle: Annotated[float, Field(ge=-360, le=360, description="Degrees counter-clockwise.")] = 0.0,
        flip: Annotated[
            str | None,
            Field(description="Flip direction: horizontal, vertical, or both."),
        ] = None,
        auto_orient: Annotated[bool, Field(description="Apply EXIF orientation first.")] = True,
        expand: Annotated[bool, Field(description="Expand canvas to prevent clipping.")] = True,
        background: Annotated[
            tuple[int, int, int, int] | None, Field(description="RGBA fill for exposed areas.")
        ] = None,
        output_format: _FormatArg = None,
        quality: _QualityArg = None,
        strip_metadata: _StripArg = None,
        output_name: _OutputNameArg = None,
        return_image: _ReturnImageArg = True,
        return_base64: _ReturnBase64Arg = False,
    ) -> ImageResult:
        with ctx.gate.slot():
            loaded = await ctx.load_input(path=path, base64_data=base64_data, url=url)
            try:
                fmt, notes = _output_spec_key(loaded, output_format)
                operations: list[Any] = []
                if auto_orient:
                    from ..imaging.pipeline import AutoOrientOperation

                    operations.append(AutoOrientOperation())
                operations.append(
                    RotateOperation(
                        angle=angle,
                        expand=expand,
                        background=background or (255, 255, 255, 0),
                    )
                )
                if flip:
                    operations.append(FlipOperation(direction=flip))  # type: ignore[arg-type]
                options = encode_options(quality=quality, strip_metadata=strip_metadata)
                outputs, pipe_notes, steps, encoded = await ctx.process(
                    loaded.image,
                    operations,
                    fmt=fmt,
                    options=options,
                    subdirectory=SUBDIR_EDIT,
                    filename_suffix="rotated",
                    source=loaded,
                    include_base64=return_base64,
                    output_name=output_name,
                )
                return ctx.builder.build_image_result(
                    operation="image_rotate",
                    outputs=outputs,
                    input_summary=_summarise_input(loaded),
                    input_bytes=loaded.byte_size,
                    notes=notes + pipe_notes + encoded.notes + loaded.notes,
                    steps=steps,
                    inline=return_image,
                )
            finally:
                loaded.close()

    # -------------------------------------------------------------- thumbnail
    @server.tool(
        name="image_thumbnail",
        title="Create a thumbnail",
        description="Create a thumbnail with attention-, entropy-, or center-based cropping.",
    )
    @tool_guard(ctx)
    async def image_thumbnail(
        path: _PathArg = None,
        base64_data: _Base64Arg = None,
        url: _UrlArg = None,
        size: Annotated[
            tuple[int, int] | None,
            Field(description="Thumbnail width and height."),
        ] = None,
        max_dimension: Annotated[
            int | None,
            Field(ge=1, le=100_000, description="Fit within a square of this size, preserving ratio."),
        ] = None,
        method: Annotated[
            str,
            Field(description="Crop method: attention, entropy, or center."),
        ] = "attention",
        sharpen: Annotated[float, Field(ge=0, le=5, description="Sharpening amount; 0 disables.")] = 0.6,
        output_format: _FormatArg = None,
        quality: _QualityArg = None,
        strip_metadata: _StripArg = None,
        output_name: _OutputNameArg = None,
        return_image: _ReturnImageArg = True,
        return_base64: _ReturnBase64Arg = False,
    ) -> ImageResult:
        with ctx.gate.slot():
            loaded = await ctx.load_input(path=path, base64_data=base64_data, url=url)
            try:
                if size is None and max_dimension is None:
                    raise InvalidArgumentError("provide either size=[w,h] or max_dimension")
                fmt, notes = _output_spec_key(loaded, output_format)

                if size is not None:
                    from ..imaging.pipeline import SmartCropOperation

                    operations: list[Any] = [
                        SmartCropOperation(width=size[0], height=size[1], method=method, scale=True)  # type: ignore[arg-type]
                    ]
                else:
                    assert max_dimension is not None
                    operations = [
                        ResizeOperation(
                            width=max_dimension,
                            height=max_dimension,
                            fit="inside",
                            only_shrink=method == "center",
                        )
                    ]

                if sharpen > 0:
                    operations.append(SharpenOperation(amount=sharpen))

                options = encode_options(quality=quality or 85, strip_metadata=strip_metadata)
                outputs, pipe_notes, steps, encoded = await ctx.process(
                    loaded.image,
                    operations,
                    fmt=fmt,
                    options=options,
                    subdirectory=SUBDIR_RESIZE,
                    filename_suffix="thumb",
                    source=loaded,
                    include_base64=return_base64,
                    output_name=output_name,
                )
                return ctx.builder.build_image_result(
                    operation="image_thumbnail",
                    outputs=outputs,
                    input_summary=_summarise_input(loaded),
                    input_bytes=loaded.byte_size,
                    notes=notes + pipe_notes + encoded.notes + loaded.notes,
                    steps=steps,
                    inline=return_image,
                )
            finally:
                loaded.close()


# --------------------------------------------------------------------- helpers


def _dimension_suffix(width: int | None, height: int | None, percent: float | None) -> str:
    if percent is not None:
        return f"{percent:g}pct"
    if width and height:
        return f"{width}x{height}"
    if width:
        return f"w{width}"
    if height:
        return f"h{height}"
    return "resized"


async def _encode_loaded(ctx: ToolContext, loaded: LoadedImage, spec, options: EncodeOptions):
    """Encode, preserving animation when both the input and target support it.

    An animated GIF converted to WebP should stay animated; converting one to
    PNG cannot, and the loader's note already records that only the first frame
    was used.
    """
    import anyio

    from ..imaging.encode import encode

    if ctx.wants_animation(loaded, spec):
        frames, durations = await anyio.to_thread.run_sync(_collect_frames, loaded)
        return await ctx.encode_frames(frames, spec, options, durations=durations, loop=0)
    return await anyio.to_thread.run_sync(encode, loaded.image, spec, options)


def _collect_frames(loaded: LoadedImage) -> tuple[list[Any], list[int]]:
    frames = []
    durations: list[int] = []
    for index in range(loaded.frames):
        loaded.image.seek(index)
        frames.append(loaded.image.convert("RGBA").copy())
        durations.append(int(loaded.image.info.get("duration", 100)))
    loaded.image.seek(0)
    return frames, durations


def _extract_exif(loaded: LoadedImage, *, max_entries: int) -> dict[str, Any] | None:
    """Decode EXIF into a JSON-safe dict."""
    from PIL import ExifTags

    raw = loaded.image.info.get("exif")
    if not raw:
        return None
    try:
        from PIL import Image as PILImage

        exif = PILImage.Exif()
        exif.load(raw)
        decoded: dict[str, Any] = {}
        for tag_id, value in list(exif.items())[:max_entries]:
            name = ExifTags.TAGS.get(tag_id, f"Tag{tag_id}")
            if isinstance(value, bytes):
                decoded[name] = f"<{len(value)} bytes>"
            elif isinstance(value, (int, float, str)):
                decoded[name] = value
            else:
                decoded[name] = str(value)[:200]
        return decoded or None
    except Exception:
        logger.debug("EXIF decode failed", exc_info=True)
        return None


def _dominant_colours(loaded: LoadedImage, *, count: int = 6) -> list[dict[str, Any]]:
    """Most common colours in the image, quantised to a small palette."""
    from PIL import Image as PILImage

    small = loaded.image.convert("RGB").copy()
    small.thumbnail((160, 160), PILImage.Resampling.BILINEAR)
    quantised = small.quantize(colors=count, method=PILImage.Quantize.MEDIANCUT)
    palette = quantised.getpalette() or []
    counts = sorted(quantised.getcolors() or [], reverse=True)
    total = sum(item[0] for item in counts) or 1
    colours: list[dict[str, Any]] = []
    for occurrences, index in counts[:count]:
        base = index * 3
        if base + 2 >= len(palette):
            continue
        red, green, blue = palette[base : base + 3]
        colours.append(
            {
                "hex": f"#{red:02x}{green:02x}{blue:02x}",
                "rgb": [red, green, blue],
                "share": round(occurrences / total, 4),
            }
        )
    return colours


async def _search_quality(
    ctx: ToolContext,
    image,
    spec,
    *,
    quality: int | None,
    target_bytes: int | None,
    strip_metadata: bool | None,
    effort: int | None,
    lossless: bool,
    deadline=None,
):
    """Pick encoder settings, binary-searching quality when a size target is set.

    ``image`` is the exact bitmap that will be encoded, not necessarily the
    original: when a caller is producing a resized variant, the search has to
    measure that variant or the chosen quality will be wrong for it.
    """
    import anyio

    notes: list[str] = []
    base = encode_options(
        quality=quality,
        lossless=lossless,
        effort=effort,
        strip_metadata=strip_metadata,
    )

    if target_bytes is None:
        if quality is None and not lossless:
            base.quality = 82
        return base, notes

    if spec.pillow_format not in {"JPEG", "WEBP", "AVIF", "JPEG2000"}:
        notes.append(f"{spec.label} is not quality-tunable; wrote it as-is and could not honour target_size_kb")
        return base, notes

    def measure(candidate: int) -> int:
        from ..imaging.encode import encode

        trial = encode_options(
            quality=candidate,
            lossless=False,
            effort=effort,
            strip_metadata=strip_metadata,
        )
        return len(encode(image, spec, trial).data)

    def search() -> tuple[int, int]:
        # Quality is monotonic in size for these codecs, so a bounded binary
        # search finds the highest quality that still fits. Five probes narrow
        # a 1-100 range to within about 3 quality points, which is finer than
        # the granularity that matters, and each probe is a full encode.
        low, high = 1, 100
        best_quality, best_size = 1, measure(1)
        if best_size > target_bytes:
            return best_quality, best_size
        for _ in range(5):
            if deadline is not None:
                # This is the main amplification path: without a budget,
                # optimize_web could run this search for every one of twelve
                # widths, for over a hundred full encodes in one call.
                deadline.check("target-size search")
            if low >= high:
                break
            middle = (low + high + 1) // 2
            size = measure(middle)
            if size <= target_bytes:
                low, best_quality, best_size = middle, middle, size
            else:
                high = middle - 1
        return best_quality, best_size

    best_quality, best_size = await anyio.to_thread.run_sync(search)
    base.quality = best_quality

    if best_size > target_bytes:
        notes.append(
            f"could not reach {human_bytes(target_bytes)} even at minimum quality; "
            f"smallest achievable is {human_bytes(best_size)} — consider resizing or a different format"
        )
    else:
        notes.append(f"searched quality {best_quality} to fit {human_bytes(target_bytes)} ({human_bytes(best_size)})")
    return base, notes


__all__ = ["register"]

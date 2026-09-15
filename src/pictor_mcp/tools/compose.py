"""Composition and bulk tools: watermark, background removal, pipeline, batch.

``image_transform`` is the workhorse: it runs an ordered list of operations in a
single call, which matters because each round trip through an agent costs more
than the pixels do. ``image_batch`` applies the same list across many files with
per-file error isolation, so one malformed input cannot fail a whole run.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import anyio
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..errors import InvalidArgumentError, PictorError
from ..imaging.encode import encode
from ..imaging.formats import OUTPUT_FORMATS, resolve_output_format
from ..imaging.loader import LoadedImage, display_name
from ..imaging.pipeline import (
    BackgroundRemoveOperation,
    FeatherOperation,
    Operation,
    WatermarkImageOperation,
    WatermarkTextOperation,
)
from ..models import BatchFileResult, BatchResult, ImageResult
from ..outputs import human_bytes
from ..security.paths import safe_filename
from .context import (
    SUBDIR_BATCH,
    SUBDIR_CUTOUT,
    SUBDIR_EDIT,
    ToolContext,
    encode_options,
    error_payload,
    tool_guard,
)

logger = logging.getLogger(__name__)


def register(server: MCPServer, ctx: ToolContext) -> None:
    """Register composition and bulk tools."""

    # -------------------------------------------------------------- transform
    @server.tool(
        name="image_transform",
        title="Apply an ordered image pipeline",
        description=(
            "Run several operations in one call, in order. Prefer this over chaining single-step "
            "tools: it transfers the image once and applies every step server-side. "
            "Operations: auto_orient, resize, crop, rotate, flip, sharpen, blur, smart_crop, "
            "watermark_text, watermark_image, background_remove, feather."
        ),
    )
    @tool_guard
    async def image_transform(
        path: Annotated[str | None, Field(description="Path to the image, inside an allowed input root.")] = None,
        base64_data: Annotated[str | None, Field(description="Inline base64 image, optionally a data: URI.")] = None,
        url: Annotated[str | None, Field(description="http(s) URL to fetch (disabled by default).")] = None,
        operations: Annotated[
            list[Operation],
            Field(min_length=1, max_length=32, description="Ordered list of operations, each with an 'op' field."),
        ] = ...,
        output_format: Annotated[
            str | None,
            Field(description="Output codec; defaults to the input format."),
        ] = None,
        quality: Annotated[int | None, Field(ge=1, le=100, description="Lossy quality for the final encode.")] = None,
        lossless: Annotated[bool, Field(description="Use a lossless encode where supported.")] = False,
        strip_metadata: Annotated[bool | None, Field(description="Strip EXIF/GPS metadata from the output.")] = None,
        output_name: Annotated[str | None, Field(description="Override the output filename.")] = None,
        return_image: Annotated[bool, Field(description="Embed the result inline for vision clients.")] = True,
        return_base64: Annotated[bool, Field(description="Include base64 in the JSON result.")] = False,
    ) -> ImageResult:
        with ctx.gate.slot():
            loaded = await ctx.load_input(path=path, base64_data=base64_data, url=url)
            try:
                fmt, notes = _output_key(loaded, output_format)
                options = encode_options(
                    quality=quality,
                    lossless=lossless,
                    strip_metadata=strip_metadata,
                )
                outputs, pipe_notes, steps, encoded = await ctx.process(
                    loaded.image,
                    operations,
                    fmt=fmt,
                    options=options,
                    subdirectory=SUBDIR_EDIT,
                    filename_suffix="transformed",
                    source=loaded,
                )
                if return_base64 and outputs:
                    outputs = await _with_base64(ctx, outputs)
                return ctx.builder.build_image_result(
                    operation="image_transform",
                    outputs=outputs,
                    input_summary=_summarise(loaded),
                    input_bytes=loaded.byte_size,
                    notes=notes + pipe_notes + encoded.notes + loaded.notes,
                    steps=steps,
                    inline=return_image,
                )
            finally:
                loaded.close()

    # -------------------------------------------------------------- watermark
    @server.tool(
        name="image_watermark",
        title="Watermark an image",
        description=(
            "Add a text or image watermark with position, opacity, rotation and optional tiling. "
            "Supply either 'text' or 'watermark_path'."
        ),
    )
    @tool_guard
    async def image_watermark(
        path: Annotated[str | None, Field(description="Path to the base image.")] = None,
        base64_data: Annotated[str | None, Field(description="Inline base64 base image.")] = None,
        url: Annotated[str | None, Field(description="http(s) URL to fetch.")] = None,
        text: Annotated[str | None, Field(max_length=500, description="Watermark text.")] = None,
        watermark_path: Annotated[
            str | None,
            Field(description="Path to a logo/watermark image (mutually exclusive with text)."),
        ] = None,
        position: Annotated[
            str,
            Field(description="center, top, bottom, left, right, or a corner like bottom-right."),
        ] = "bottom-right",
        opacity: Annotated[float, Field(ge=0, le=1, description="Watermark opacity.")] = 0.6,
        font_size: Annotated[
            int | None,
            Field(ge=4, le=4096, description="Text size in pixels; defaults to a fraction of the image."),
        ] = None,
        font_family: Annotated[
            str | None,
            Field(description="Font name from the server's font index, e.g. DejaVuSans-Bold."),
        ] = None,
        colour: Annotated[tuple[int, int, int], Field(description="Text RGB colour.")] = (255, 255, 255),
        padding: Annotated[int, Field(ge=0, le=10_000, description="Distance from the edge in pixels.")] = 16,
        tile: Annotated[bool, Field(description="Repeat the watermark across the whole image.")] = False,
        spacing: Annotated[int, Field(ge=0, le=10_000, description="Gap between tiles in pixels.")] = 0,
        rotation: Annotated[float, Field(ge=-180, le=180, description="Rotate the watermark, in degrees.")] = 0.0,
        shadow: Annotated[bool, Field(description="Draw a drop shadow behind text for legibility.")] = True,
        stroke_width: Annotated[int, Field(ge=0, le=32, description="Outline width for text.")] = 0,
        scale: Annotated[
            float | None,
            Field(gt=0, le=4, description="Logo width as a fraction of the image's shorter edge."),
        ] = None,
        output_format: Annotated[str | None, Field(description="Output codec; defaults to the input format.")] = None,
        quality: Annotated[int | None, Field(ge=1, le=100, description="Lossy quality.")] = None,
        strip_metadata: Annotated[bool | None, Field(description="Strip metadata from the output.")] = None,
        output_name: Annotated[str | None, Field(description="Override the output filename.")] = None,
        return_image: Annotated[bool, Field(description="Embed the result inline.")] = True,
        return_base64: Annotated[bool, Field(description="Include base64 in the JSON result.")] = False,
    ) -> ImageResult:
        with ctx.gate.slot():
            if bool(text) == bool(watermark_path):
                raise InvalidArgumentError("provide exactly one of 'text' or 'watermark_path'")
            loaded = await ctx.load_input(path=path, base64_data=base64_data, url=url)
            try:
                fmt, notes = _output_key(loaded, output_format)
                if text:
                    operation: Any = WatermarkTextOperation(
                        text=text,
                        position=position,
                        opacity=opacity,
                        font_size=font_size,
                        font_family=font_family,
                        colour=colour,
                        padding=padding,
                        tile=tile,
                        spacing=spacing,
                        rotation=rotation,
                        shadow=shadow,
                        stroke_width=stroke_width,
                    )
                else:
                    assert watermark_path is not None
                    operation = WatermarkImageOperation(
                        path=watermark_path,
                        position=position,
                        opacity=opacity,
                        scale=scale,
                        padding=padding,
                        tile=tile,
                        spacing=spacing,
                    )
                options = encode_options(quality=quality, strip_metadata=strip_metadata)
                outputs, pipe_notes, steps, encoded = await ctx.process(
                    loaded.image,
                    [operation],
                    fmt=fmt,
                    options=options,
                    subdirectory=SUBDIR_EDIT,
                    filename_suffix="watermarked",
                    source=loaded,
                )
                if return_base64 and outputs:
                    outputs = await _with_base64(ctx, outputs)
                return ctx.builder.build_image_result(
                    operation="image_watermark",
                    outputs=outputs,
                    input_summary=_summarise(loaded),
                    input_bytes=loaded.byte_size,
                    notes=notes + pipe_notes + encoded.notes + loaded.notes,
                    steps=steps,
                    inline=return_image,
                )
            finally:
                loaded.close()

    # ------------------------------------------------------- background removal
    @server.tool(
        name="image_background_remove",
        title="Remove an image background",
        description=(
            "Make the background transparent. method='color' keys out a flat border colour "
            "(instant, offline, best for product/logo shots); method='ml' uses a U^2-Net model "
            "for arbitrary backgrounds; method='auto' picks ML when the model is installed."
        ),
    )
    @tool_guard
    async def image_background_remove(
        path: Annotated[str | None, Field(description="Path to the image.")] = None,
        base64_data: Annotated[str | None, Field(description="Inline base64 image.")] = None,
        url: Annotated[str | None, Field(description="http(s) URL to fetch.")] = None,
        method: Annotated[str, Field(description="auto, color or ml.")] = "auto",
        tolerance: Annotated[
            float, Field(ge=0, le=441, description="Colour distance treated as background (color method).")
        ] = 32.0,
        softness: Annotated[
            float, Field(ge=0, le=441, description="Width of the soft alpha ramp at the edges.")
        ] = 24.0,
        edge_connected: Annotated[
            bool,
            Field(description="Only remove background regions touching the border, protecting interior matches."),
        ] = True,
        model: Annotated[
            str, Field(max_length=64, description="ML model name: u2net, u2netp, isnet-general-use, silueta.")
        ] = "u2net",
        feather: Annotated[
            float, Field(ge=0, le=100, description="Soften the resulting alpha edge by this many pixels.")
        ] = 0.0,
        background: Annotated[
            tuple[int, int, int] | None,
            Field(description="Composite onto this RGB colour instead of leaving transparency."),
        ] = None,
        output_format: Annotated[
            str | None,
            Field(description="Output codec; must support alpha (png, webp, avif, tiff)."),
        ] = "png",
        strip_metadata: Annotated[bool | None, Field(description="Strip metadata from the output.")] = None,
        output_name: Annotated[str | None, Field(description="Override the output filename.")] = None,
        return_image: Annotated[bool, Field(description="Embed the result inline.")] = True,
        return_base64: Annotated[bool, Field(description="Include base64 in the JSON result.")] = False,
    ) -> ImageResult:
        with ctx.gate.slot():
            loaded = await ctx.load_input(path=path, base64_data=base64_data, url=url)
            try:
                key, notes = _output_key(loaded, output_format)
                spec = resolve_output_format(key)
                if background is None and not spec.supports_alpha:
                    raise InvalidArgumentError(
                        f"{spec.label} cannot store transparency; choose a format with alpha "
                        "(png, webp, avif, tiff) or pass 'background' to composite onto a colour"
                    )
                operations: list[Any] = [
                    BackgroundRemoveOperation(
                        method=method,  # type: ignore[arg-type]
                        tolerance=tolerance,
                        softness=softness,
                        edge_connected=edge_connected,
                        model=model,
                        background=(*background, 255) if background else None,
                    )
                ]
                if feather > 0:
                    operations.append(FeatherOperation(radius=feather))
                options = encode_options(strip_metadata=strip_metadata)
                outputs, pipe_notes, steps, encoded = await ctx.process(
                    loaded.image,
                    operations,
                    fmt=key,
                    options=options,
                    subdirectory=SUBDIR_CUTOUT,
                    filename_suffix="cutout",
                    source=loaded,
                )
                if return_base64 and outputs:
                    outputs = await _with_base64(ctx, outputs)
                return ctx.builder.build_image_result(
                    operation="image_background_remove",
                    outputs=outputs,
                    input_summary=_summarise(loaded),
                    input_bytes=loaded.byte_size,
                    notes=notes + pipe_notes + encoded.notes + loaded.notes,
                    steps=steps,
                    inline=return_image,
                )
            finally:
                loaded.close()

    # ------------------------------------------------------------------ batch
    @server.tool(
        name="image_batch",
        title="Apply a pipeline to many images",
        description=(
            "Run the same operation list across every image in a directory, a glob, or an explicit "
            "list of paths. Each file is independent: a failure is reported per file and does not "
            "abort the run. Use a {name} placeholder in output_name for per-file suffixes."
        ),
    )
    @tool_guard
    async def image_batch(
        directory: Annotated[
            str | None,
            Field(description="Directory (inside an input root) to process recursively."),
        ] = None,
        glob: Annotated[
            str | None,
            Field(description="Glob pattern inside an input root, e.g. 'photos/**/*.jpg'."),
        ] = None,
        paths: Annotated[
            list[str] | None,
            Field(max_length=256, description="Explicit list of input paths."),
        ] = None,
        operations: Annotated[
            list[Operation],
            Field(min_length=1, max_length=32, description="Ordered operations applied to every file."),
        ] = ...,
        output_format: Annotated[
            str | None,
            Field(description="Output codec; defaults to each input's own format."),
        ] = None,
        quality: Annotated[int | None, Field(ge=1, le=100, description="Lossy quality for every output.")] = None,
        strip_metadata: Annotated[bool | None, Field(description="Strip metadata from every output.")] = None,
        limit: Annotated[
            int | None,
            Field(ge=1, le=10_000, description="Maximum files to process in this call."),
        ] = None,
        return_first_image: Annotated[
            bool, Field(description="Embed only the first output inline, to bound payload size.")
        ] = False,
    ) -> BatchResult:
        with ctx.gate.slot():
            targets = await anyio.to_thread.run_sync(
                lambda: _collect_targets(ctx, directory=directory, glob=glob, paths=paths, limit=limit)
            )
            if not targets:
                raise InvalidArgumentError(
                    "no input files matched; check the directory, glob or paths and the configured input roots"
                )

            results: list[BatchFileResult] = []
            total_in = 0
            total_out = 0
            first_outputs = None
            succeeded = 0

            for target in targets:
                entry = await _process_one(
                    ctx,
                    target,
                    operations=operations,
                    output_format=output_format,
                    quality=quality,
                    strip_metadata=strip_metadata,
                )
                results.append(entry)
                if entry.status == "ok":
                    succeeded += 1
                    total_in += entry.size_change.input_bytes or 0 if entry.size_change else 0
                    total_out += entry.size_change.output_bytes or 0 if entry.size_change else 0
                    if first_outputs is None:
                        first_outputs = entry.outputs

            failed = sum(1 for item in results if item.status == "failed")
            report = BatchResult(
                processed=len(results),
                succeeded=succeeded,
                failed=failed,
                totalInputBytes=total_in,
                totalOutputBytes=total_out,
                files=results,
                notes=[
                    f"processed {len(results)} files: {succeeded} succeeded, {failed} failed",
                ],
            )
            inline_bytes = None
            inline_mime = None
            if return_first_image and first_outputs:
                primary = first_outputs[0]
                try:
                    data = await anyio.to_thread.run_sync(
                        partial(
                            ctx.jail.read_output_bytes,
                            primary.path,
                            max_bytes=ctx.config.limits.max_inline_image_bytes,
                        )
                    )
                    inline_bytes = data
                    inline_mime = primary.mime_type
                    report.inline_image_included = True
                except PictorError:
                    logger.debug("could not re-read batch output for inlining", exc_info=True)

            from ..outputs import build_result

            text = "\n".join(
                [
                    f"Batch complete: {succeeded}/{len(results)} succeeded" + (f", {failed} failed" if failed else ""),
                    f"Total: {human_bytes(total_in)} -> {human_bytes(total_out)}",
                    *[f"  {item.source}: {item.status}" for item in results[:20]],
                    *(["  ..."] if len(results) > 20 else []),
                ]
            )
            return build_result(report, text, inline_bytes=inline_bytes, inline_mime=inline_mime)


# --------------------------------------------------------------------- helpers


def _output_key(loaded: LoadedImage, requested: str | None) -> tuple[str, list[str]]:
    if requested:
        return requested, []
    if loaded.spec.key in OUTPUT_FORMATS:
        return loaded.spec.key, []
    return "png", [f"{loaded.spec.label} is read-only; wrote PNG instead"]


def _summarise(loaded: LoadedImage) -> dict[str, Any]:
    return {
        "width": loaded.geometry.width,
        "height": loaded.geometry.height,
        "format": loaded.original_format,
        "mode": loaded.image.mode,
        "byteSize": loaded.byte_size,
        "frames": loaded.frames,
        "hasAlpha": loaded.has_alpha,
    }


async def _with_base64(ctx: ToolContext, outputs):
    """Re-store outputs with inline base64 included.

    Re-storing would duplicate files, so the payload is attached to the existing
    descriptor instead.
    """
    import base64

    for output in outputs:
        if output.file.base64 is None:
            output.file.base64 = base64.b64encode(output.data).decode()
    return outputs


def _collect_targets(
    ctx: ToolContext,
    *,
    directory: str | None,
    glob: str | None,
    paths: list[str] | None,
    limit: int | None,
) -> list[Path]:
    """Resolve a batch selection, confined to the input roots."""
    cap = min(limit or ctx.config.limits.max_batch_files, ctx.config.limits.max_batch_files)
    jail = ctx.jail
    found: list[Path] = []

    provided = [bool(directory), bool(glob), bool(paths)]
    if sum(provided) != 1:
        raise InvalidArgumentError("provide exactly one of directory, glob or paths")

    if paths:
        for entry in paths[:cap]:
            found.append(jail.resolve_read(entry))
        return found

    if directory:
        root = jail.resolve_read(directory)
        if not root.is_dir():
            raise InvalidArgumentError("directory does not exist or is not a directory")
        candidates = sorted(p for p in root.rglob("*") if p.is_file())
    else:
        assert glob is not None
        if glob.startswith("/") or ".." in Path(glob).parts:
            raise InvalidArgumentError("glob must be relative and must not contain '..'")
        base = ctx.config.input_roots[0]
        candidates = sorted(p for p in base.glob(glob) if p.is_file())

    for candidate in candidates:
        if len(found) >= cap:
            break
        try:
            found.append(jail.resolve_read(candidate))
        except PictorError:
            # Not a readable image location; skip rather than fail the batch.
            logger.debug("skipping %s during batch collection", candidate)
    return found


async def _process_one(
    ctx: ToolContext,
    target: Path,
    *,
    operations: list[Operation],
    output_format: str | None,
    quality: int | None,
    strip_metadata: bool | None,
) -> BatchFileResult:
    """Process a single batch entry, isolating failures."""
    source_label = display_name(str(target))
    try:
        loaded = await ctx.load_input(path=str(target))
    except PictorError as exc:
        return BatchFileResult(source=source_label, status="skipped", error=error_payload(exc))
    except Exception:  # pragma: no cover - defensive
        # The detail goes to the log, not to the client. An OSError stringifies
        # to the absolute path it failed on, which would hand a remote caller
        # the server's directory layout - exactly what the single-call error
        # path already refuses to do.
        logger.exception("unexpected failure loading %s", source_label)
        return BatchFileResult(
            source=source_label,
            status="failed",
            error={"code": "internal_error", "message": "the file could not be processed"},
        )

    try:
        fmt, notes = _output_key(loaded, output_format)
        spec = resolve_output_format(fmt)
        overlays = await ctx.load_overlays(operations)
        options = encode_options(quality=quality, strip_metadata=strip_metadata)

        def work():
            from ..imaging.pipeline import apply_operations

            outcome = apply_operations(
                loaded.image,
                list(operations),
                config=ctx.config,
                fonts=ctx.fonts,
                overlays=overlays,
                resampler_for=ctx.registry.resampler_for,
            )
            return encode(outcome.image, spec, options)

        encoded = await anyio.to_thread.run_sync(work)
        filename = safe_filename(Path(source_label).stem, spec.extension)
        stored = await ctx.store_encoded(encoded, subdirectory=SUBDIR_BATCH, filename=filename)

        from ..outputs import size_change_for

        return BatchFileResult(
            source=source_label,
            status="ok",
            outputs=[stored.file],
            notes=notes + encoded.notes + loaded.notes,
            sizeChange=size_change_for(loaded.byte_size, encoded.byte_size),
        )
    except PictorError as exc:
        return BatchFileResult(source=source_label, status="failed", error=error_payload(exc))
    except Exception:  # pragma: no cover - defensive
        logger.exception("unexpected failure processing %s", source_label)
        return BatchFileResult(
            source=source_label,
            status="failed",
            error={"code": "internal_error", "message": "the operation could not be completed"},
        )
    finally:
        loaded.close()


__all__ = ["register"]

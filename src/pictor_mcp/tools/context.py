"""Shared plumbing for the tool layer.

Every tool follows the same shape: confine the input, decode it off the event
loop, transform it off the event loop, encode it, store it, and describe it.
That common path lives here so individual tools stay small enough to read and
so the security-relevant steps cannot be accidentally skipped in one tool but
not another.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from mcp_types import CallToolResult, TextContent
from PIL import Image

from .. import __version__
from ..backends import BackendRegistry
from ..config import Config
from ..errors import PictorError
from ..imaging.encode import EncodedImage, EncodeOptions, encode, encode_animation
from ..imaging.fonts import FontIndex
from ..imaging.formats import FormatSpec, resolve_output_format
from ..imaging.loader import FramePolicy, ImageLoader, ImageSource, LoadedImage
from ..imaging.pipeline import Operation, apply_operations, watermark_paths
from ..outputs import ResultBuilder, StoredOutput, default_filename
from ..security.paths import PathJail, safe_filename
from ..security.ratelimit import ConcurrencyGate, Deadline

logger = logging.getLogger(__name__)

#: Subdirectory of the output root used by each family of operations.
SUBDIR_CONVERT = "converted"
SUBDIR_RESIZE = "resized"
SUBDIR_COMPRESS = "compressed"
SUBDIR_EDIT = "edited"
SUBDIR_BATCH = "batch"
SUBDIR_WEB = "web"
SUBDIR_CUTOUT = "cutout"


@dataclass(slots=True)
class ToolContext:
    """Everything a tool needs, assembled once at startup."""

    config: Config
    jail: PathJail
    loader: ImageLoader
    builder: ResultBuilder
    fonts: FontIndex
    registry: BackendRegistry
    gate: ConcurrencyGate
    #: Reported by image_capabilities; defaults to the package version so there
    #: is only one place to change it.
    server_version: str = __version__

    # ------------------------------------------------------------------ input
    async def load_input(
        self,
        *,
        path: str | None = None,
        base64_data: str | None = None,
        url: str | None = None,
        frame_policy: FramePolicy = "first",
    ) -> LoadedImage:
        """Resolve and decode exactly one input source."""
        source = ImageSource.parse(path=path, base64_data=base64_data, url=url)
        return await self.loader.load(source, frame_policy=frame_policy)

    async def load_overlays(self, operations: Sequence[Operation]) -> dict[str, Image.Image]:
        """Pre-load every watermark image a pipeline references.

        Done in the async layer so the pipeline itself can stay synchronous and
        run entirely inside a worker thread.
        """
        overlays: dict[str, Image.Image] = {}
        for path in watermark_paths(list(operations)):
            if path in overlays:
                continue
            loaded = await self.load_input(path=path)
            overlays[path] = loaded.image.convert("RGBA")
        return overlays

    # --------------------------------------------------------------- process
    async def run_pipeline(
        self,
        image: Image.Image,
        operations: Sequence[Operation],
    ) -> tuple[Image.Image, list[str], list[str]]:
        """Apply operations on a worker thread."""
        overlays = await self.load_overlays(operations)

        deadline = self.deadline()

        def work() -> tuple[Image.Image, list[str], list[str]]:
            outcome = apply_operations(
                image,
                list(operations),
                config=self.config,
                fonts=self.fonts,
                overlays=overlays,
                resampler_for=self.registry.resampler_for,
                deadline=deadline,
            )
            return outcome.image, outcome.notes, outcome.steps

        return await anyio.to_thread.run_sync(work)

    def output_filename(
        self,
        requested: str | None,
        spec: FormatSpec,
        source: LoadedImage | None,
    ) -> str:
        """Build the output filename, forcing the extension to match the codec.

        A caller-supplied ``output_name`` is sanitised and its extension is
        replaced with the codec's own. That closes a stored-XSS path: without
        it, a caller could name a JPEG ``pwn.html``, and the signed file
        endpoint would then serve those bytes as ``text/html`` on the server's
        own origin. The extension is a property of the encoder, not a label the
        caller gets to choose.
        """
        if not requested:
            if source is not None:
                return default_filename(source, spec)
            return f"image{spec.extension}"
        return safe_filename(Path(requested).stem, spec.extension)

    def encode(
        self,
        image: Image.Image,
        fmt: str,
        options: EncodeOptions | None = None,
    ) -> tuple[EncodedImage, FormatSpec]:
        """Resolve a format name and encode (synchronous; call in a thread)."""
        spec = resolve_output_format(fmt)
        return encode(image, spec, options), spec

    async def process(
        self,
        image: Image.Image,
        operations: Sequence[Operation],
        *,
        fmt: str,
        options: EncodeOptions | None = None,
        subdirectory: str,
        filename_suffix: str = "",
        source: LoadedImage | None = None,
        include_base64: bool = False,
        output_name: str | None = None,
    ) -> tuple[list[StoredOutput], list[str], list[str], EncodedImage]:
        """Apply operations, encode, and store - the standard tool tail."""
        overlays = await self.load_overlays(operations)

        deadline = self.deadline()

        def work() -> tuple[EncodedImage, list[str], list[str]]:
            outcome = apply_operations(
                image,
                list(operations),
                config=self.config,
                fonts=self.fonts,
                overlays=overlays,
                resampler_for=self.registry.resampler_for,
                deadline=deadline,
            )
            spec = resolve_output_format(fmt)
            encoded = encode(outcome.image, spec, options)
            return encoded, outcome.notes, outcome.steps

        encoded, notes, steps = await anyio.to_thread.run_sync(work)

        if output_name:
            filename = self.output_filename(output_name, encoded.spec, source)
        else:
            filename = (
                default_filename(source, encoded.spec, suffix=filename_suffix)
                if source is not None
                else f"image{('-' + filename_suffix) if filename_suffix else ''}{encoded.spec.extension}"
            )
        stored = await self.store_encoded(
            encoded,
            subdirectory=subdirectory,
            filename=filename,
            include_base64=include_base64,
        )
        return [stored], notes, steps, encoded

    async def store_encoded(
        self,
        encoded: EncodedImage,
        *,
        subdirectory: str,
        filename: str,
        include_base64: bool = False,
    ) -> StoredOutput:
        return await anyio.to_thread.run_sync(
            lambda: self.builder.store(
                encoded,
                subdirectory=subdirectory,
                filename=filename,
                include_base64=include_base64,
            )
        )

    async def store_many(
        self,
        items: Sequence[tuple[EncodedImage, str]],
        *,
        subdirectory: str,
        include_base64: bool = False,
    ) -> list[StoredOutput]:
        """Encode/store several variants, keeping a stable order."""
        results: list[StoredOutput] = []
        for encoded, filename in items:
            results.append(
                await self.store_encoded(
                    encoded,
                    subdirectory=subdirectory,
                    filename=filename,
                    include_base64=include_base64,
                )
            )
        return results

    # ----------------------------------------------------------- animation
    def wants_animation(self, loaded: LoadedImage, spec: FormatSpec) -> bool:
        """True when the input is animated and the target format can keep it."""
        return loaded.is_animated and spec.supports_animation and loaded.frames > 1

    async def encode_frames(
        self,
        frames: list[Image.Image],
        spec: FormatSpec,
        options: EncodeOptions,
        *,
        durations: list[int] | None,
        loop: int,
    ) -> EncodedImage:
        return await anyio.to_thread.run_sync(
            lambda: encode_animation(frames, spec, options, durations=durations, loop=loop)
        )

    # --------------------------------------------------------------- guards
    def deadline(self) -> Deadline:
        """A fresh wall-clock budget for one tool call."""
        return Deadline(self.config.limits.op_timeout_seconds)

    def slot(self):
        """Concurrency slot; use as a context manager around a whole tool call."""
        return self.gate.slot()


def encode_options(
    *,
    quality: int | None = None,
    lossless: bool = False,
    progressive: bool = False,
    optimize: bool = True,
    compress_level: int | None = None,
    effort: int | None = None,
    subsampling: str | None = None,
    background: tuple[int, int, int] | None = None,
    strip_metadata: bool | None = None,
    keep_icc: bool = False,
    dpi: tuple[int, int] | None = None,
    compression: str | None = None,
) -> EncodeOptions:
    """Build :class:`EncodeOptions`, applying the server default for metadata.

    ``background`` is the RGB colour that transparency is flattened onto when
    the target codec has no alpha channel. It is deliberately a triple: an alpha
    component here would be meaningless, because the whole point is to *replace*
    transparency with an opaque colour.
    """
    resolved_background = tuple(background) if background is not None else (255, 255, 255)
    return EncodeOptions(
        quality=quality,
        lossless=lossless,
        progressive=progressive,
        optimize=optimize,
        compress_level=compress_level,
        effort=effort,
        subsampling=subsampling,
        background=resolved_background,  # type: ignore[arg-type]
        strip_metadata=strip_metadata if strip_metadata is not None else True,
        keep_icc=keep_icc,
        dpi=dpi,
        compression=compression,
        quality_layers=None,
        irreversible=True,
    )


def error_payload(exc: PictorError) -> dict[str, Any]:
    """Machine-readable error body for batch entries."""
    return exc.to_dict()


def error_result(exc: PictorError) -> CallToolResult:
    """Build an MCP error result carrying both prose and a stable code.

    The text block is what a model reads; ``structuredContent`` is what a
    programmatic caller branches on. Both are safe to expose: messages never
    contain host paths or library internals.
    """
    return CallToolResult(
        content=[TextContent(type="text", text=exc.message)],
        structuredContent={"ok": False, "error": exc.to_dict()},
        is_error=True,
    )


def tool_guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn expected failures into structured MCP errors.

    Applied under ``@server.tool`` so the SDK still introspects the original
    signature (``functools.wraps`` preserves ``__wrapped__`` and
    ``__annotations__``, which is what schema generation reads).

    Anything not derived from :class:`~pictor_mcp.errors.PictorError` is a bug,
    not a user error: it is logged with a traceback and reported to the client
    as a neutral internal failure, never with its original text, which could
    disclose a filesystem path or a library detail.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except PictorError as exc:
            logger.info("tool %s rejected the request: [%s] %s", fn.__name__, exc.code, exc.message)
            return error_result(exc)
        except Exception:
            logger.exception("tool %s crashed", fn.__name__)
            return error_result(PictorError("the operation could not be completed due to an internal error"))

    return wrapper


__all__ = [
    "SUBDIR_BATCH",
    "SUBDIR_COMPRESS",
    "SUBDIR_CONVERT",
    "SUBDIR_CUTOUT",
    "SUBDIR_EDIT",
    "SUBDIR_RESIZE",
    "SUBDIR_WEB",
    "ToolContext",
    "encode_options",
    "error_payload",
    "error_result",
    "tool_guard",
]

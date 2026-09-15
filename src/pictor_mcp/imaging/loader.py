"""Input resolution and safe decoding.

Three input shapes are accepted, in increasing order of risk:

``path``
    A filesystem path confined by :class:`~pictor_mcp.security.paths.PathJail`.
    The bytes are read through a descriptor that cannot be redirected by a
    symlink swap, and the size cap is enforced while reading.
``base64``
    An inline payload, optionally a ``data:`` URI. Capped before decoding, and
    decoded with ``validate=True`` so silently-skipped garbage is an error.
``url``
    Fetched through :class:`~pictor_mcp.security.net.SafeFetcher`, which is
    disabled unless explicitly enabled.

Whatever the source, the bytes are treated as hostile until Pillow has named
the format and that name is on the allow-list. The extension, the caller's
``media_type`` hint and the HTTP ``Content-Type`` are advisory only - they are
never used to choose a decoder.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import anyio
from PIL import Image, UnidentifiedImageError

from ..config import Config
from ..errors import InvalidArgumentError, LimitExceededError, UnsupportedFormatError
from ..security.limits import Geometry, check_base64_size, check_decoded_geometry
from ..security.net import SafeFetcher
from ..security.paths import PathJail
from .formats import FormatSpec, assert_decoded_format_allowed

logger = logging.getLogger(__name__)

SourceKind = Literal["path", "base64", "url"]

#: How to treat multi-frame inputs.
FramePolicy = Literal["first", "all"]

_DATA_URI_PREFIX = "data:"


@dataclass(slots=True)
class ImageSource:
    """A requested input, before resolution."""

    kind: SourceKind
    value: str

    @classmethod
    def parse(
        cls,
        *,
        path: str | None = None,
        base64_data: str | None = None,
        url: str | None = None,
    ) -> ImageSource:
        """Pick exactly one input channel."""
        provided = [(k, v) for k, v in (("path", path), ("base64", base64_data), ("url", url)) if v]
        if not provided:
            raise InvalidArgumentError("no image supplied: pass exactly one of 'path', 'base64' or 'url'")
        if len(provided) > 1:
            raise InvalidArgumentError(
                f"supply only one of 'path', 'base64' or 'url' (got {', '.join(k for k, _ in provided)})"
            )
        kind, value = provided[0]
        return cls(kind=kind, value=value)  # type: ignore[arg-type]

    def redacted(self) -> dict[str, str]:
        """A log-safe description that never echoes inline image bytes."""
        if self.kind == "base64":
            return {"kind": "base64", "bytes": str(len(self.value))}
        return {"kind": self.kind, "value": self.value}


@dataclass(slots=True)
class LoadedImage:
    """A decoded image plus the provenance a caller needs to reason about it."""

    image: Image.Image
    spec: FormatSpec
    source: ImageSource
    byte_size: int
    frames: int
    geometry: Geometry
    mime_type: str
    original_format: str
    #: Non-fatal notes worth surfacing (e.g. "animation flattened to first frame").
    notes: list[str] = field(default_factory=list)

    @property
    def has_alpha(self) -> bool:
        return self.image.mode in {"RGBA", "LA", "PA"} or "transparency" in self.image.info

    @property
    def is_animated(self) -> bool:
        return self.frames > 1

    def close(self) -> None:
        try:
            self.image.close()
        except Exception:  # pragma: no cover - closing is best effort
            logger.debug("failed to close image", exc_info=True)


class ImageLoader:
    """Resolves and decodes image inputs under the configured limits."""

    __slots__ = ("_config", "_fetcher", "_jail")

    def __init__(self, config: Config, jail: PathJail, fetcher: SafeFetcher) -> None:
        self._config = config
        self._jail = jail
        self._fetcher = fetcher

    # ------------------------------------------------------------- acquisition
    async def acquire_bytes(self, source: ImageSource) -> tuple[bytes, str]:
        """Return ``(bytes, origin_label)`` for a source."""
        limits = self._config.limits
        if source.kind == "path":
            data = await anyio.to_thread.run_sync(
                partial(self._jail.read_bytes, source.value, max_bytes=limits.max_file_bytes)
            )
            return data, f"file:{display_name(source.value)}"
        if source.kind == "base64":
            return self._decode_base64(source.value), "inline"
        return await self._fetch(source.value)

    def _decode_base64(self, raw: str) -> bytes:
        payload = raw.strip()
        if payload.startswith(_DATA_URI_PREFIX):
            _, _, payload = payload.partition(",")
        # Tolerate whitespace/newlines from models that pretty-print base64.
        payload = "".join(payload.split())
        limits = self._config.limits
        check_base64_size(payload, limits)
        try:
            # validate=True turns "ignored non-base64 characters" into an error
            # instead of silently accepting a truncated payload.
            return base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidArgumentError("inline image is not valid base64") from exc

    async def _fetch(self, url: str) -> tuple[bytes, str]:
        if not self._config.fetch.enabled:
            from ..errors import NetworkDisabledError

            raise NetworkDisabledError(
                "fetching images by URL is disabled; set PICTOR_ALLOW_NET_FETCH=true to enable it"
            )

        result = await anyio.to_thread.run_sync(self._fetcher.fetch, url)
        return result.data, f"url:{result.final_url}"

    # ---------------------------------------------------------------- decoding
    async def load(self, source: ImageSource, *, frame_policy: FramePolicy = "first") -> LoadedImage:
        data, origin = await self.acquire_bytes(source)
        # Decoding is CPU-bound and can take seconds on a large image; running
        # it inline would stall every other client's request on the event loop.
        return await anyio.to_thread.run_sync(
            partial(
                self.decode_bytes,
                data,
                source=source,
                origin=origin,
                frame_policy=frame_policy,
            )
        )

    def decode_bytes(
        self,
        data: bytes,
        *,
        source: ImageSource,
        origin: str,
        frame_policy: FramePolicy = "first",
    ) -> LoadedImage:
        """Decode already-acquired bytes with the full validation chain."""
        limits = self._config.limits
        if not data:
            raise InvalidArgumentError("image data is empty")

        try:
            # Pillow reads the header lazily; geometry is validated before any
            # pixel buffer is allocated.
            image = Image.open(io.BytesIO(data))
        except UnidentifiedImageError as exc:
            raise UnsupportedFormatError(
                "the supplied bytes are not a recognisable image",
            ) from exc
        except Image.DecompressionBombError as exc:
            raise LimitExceededError(
                "image exceeds the decoder's decompression-bomb threshold",
            ) from exc
        except (OSError, ValueError, SyntaxError) as exc:
            raise UnsupportedFormatError("image header could not be parsed") from exc

        spec = assert_decoded_format_allowed(image.format, source="input")
        self._note_origin(image, origin)

        try:
            width, height = image.size
            # Per-frame geometry is checked BEFORE the frame table is walked.
            # Counting frames means seeking, and for APNG the decoder loads the
            # previous frame on each seek - so with the checks the other way
            # round, up to `max_frames` frames in the (max_pixels, 2 *
            # max_pixels] window would be fully decoded and discarded before the
            # pixel ceiling ever applied.
            check_decoded_geometry(Geometry(width=width, height=height), limits, context="input")

            frames = self._count_frames(image, limits.max_frames, frame_policy)
            geometry = Geometry(width=width, height=height, frames=frames)
            check_decoded_geometry(geometry, limits, context="input")

            if frame_policy == "first" and frames > 1:
                image.seek(0)

            # Force the decode now, inside our validation frame, so a decoder
            # failure surfaces here rather than during an operation.
            image.load()
        except Exception:
            image.close()
            raise

        notes: list[str] = []
        if frames > 1 and frame_policy == "first":
            notes.append(f"input has {frames} frames; only the first was used")

        return LoadedImage(
            image=image,
            spec=spec,
            source=source,
            byte_size=len(data),
            frames=frames,
            geometry=geometry,
            mime_type=spec.mime,
            original_format=spec.key,
            notes=notes,
        )

    @staticmethod
    def _count_frames(image: Image.Image, max_frames: int, frame_policy: FramePolicy) -> int:
        """Count frames, aborting as soon as the ceiling is passed.

        The count is needed even when only the first frame will be used, because
        ``image_info`` reports it and because a hostile frame table should be
        rejected rather than silently ignored.
        """
        del frame_policy  # the ceiling applies to every policy
        if not getattr(image, "is_animated", False):
            return 1
        count = 1
        while True:
            try:
                image.seek(count)
            except EOFError:
                break
            except (OSError, ValueError) as exc:
                raise UnsupportedFormatError("animation frame table is corrupt") from exc
            count += 1
            if count > max_frames:
                raise LimitExceededError(
                    f"animation exceeds the {max_frames} frame limit",
                    limit=max_frames,
                )
        image.seek(0)
        return count

    @staticmethod
    def _note_origin(image: Image.Image, origin: str) -> None:
        """Attach provenance for later reporting, without trusting it."""
        image.info.setdefault("pictor_origin", origin)

    # ----------------------------------------------------------------- helpers
    def decode_output_bytes(self, data: bytes, *, expected: FormatSpec) -> Image.Image:
        """Decode bytes we produced ourselves; still validated, never assumed."""
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as exc:  # pragma: no cover - our own encoder output
            raise UnsupportedFormatError("encoded output could not be re-read") from exc
        return image

    def metadata_for(self, loaded: LoadedImage) -> dict[str, Any]:
        """Summarise an image for tool results."""
        return {
            "width": loaded.geometry.width,
            "height": loaded.geometry.height,
            "frames": loaded.frames,
            "mode": loaded.image.mode,
            "format": loaded.original_format,
            "mimeType": loaded.mime_type,
            "byteSize": loaded.byte_size,
            "hasAlpha": loaded.has_alpha,
            "isAnimated": loaded.is_animated,
            "source": loaded.source.redacted(),
        }


def display_name(value: str) -> str:
    """Shorten a caller-supplied path to its final component.

    Error messages and results must never disclose the server's directory
    layout, so only the basename is echoed back.
    """
    import os

    return os.path.basename(value.rstrip("/")) or value


__all__ = ["FramePolicy", "ImageLoader", "ImageSource", "LoadedImage", "SourceKind", "display_name"]

"""Encoding and metadata hygiene.

Two things make this more than a thin ``image.save()`` wrapper.

**Output modes.** Callers ask for "WebP" from a CMYK TIFF with an alpha
channel, and every codec fails differently. Modes are coerced deliberately:
alpha is composited onto a known background rather than silently turning black,
CMYK is converted through a real colour transform, and palette images are
quantised once with a stated method.

**Metadata.** A resized photo still carries the original GPS coordinates, the
camera serial number and any embedded thumbnail. Metadata is therefore stripped
by default: privacy-preserving, smaller, and it removes a whole class of parser
bugs from anything downstream. It is opt-in per call when a caller genuinely
needs the EXIF.

Every codec's keyword arguments are the ones the installed Pillow actually
reads. Options a codec does not understand are dropped rather than passed
through, because Pillow silently ignores unknown keywords for some formats and
raises for others - silently ignoring is the worse failure, since the caller
believes the quality setting took effect.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from ..errors import InvalidArgumentError, UnsupportedFormatError
from .formats import FormatSpec

logger = logging.getLogger(__name__)

#: ``image.info`` keys removed when metadata stripping is on. ``transparency``,
#: ``duration`` and ``loop`` are deliberately absent: they are pixel-affecting,
#: not metadata, and dropping them corrupts GIF/PNG output.
_METADATA_KEYS = frozenset(
    {
        "exif",
        "icc_profile",
        "pnginfo",
        "xmp",
        "XML:com.adobe.xmp",
        "comment",
        "Comment",
        "dpi",
        "jfif",
        "jfif_version",
        "jfif_unit",
        "jfif_density",
        "adobe",
        "adobe_transform",
        "photoshop",
        "iptc",
        "pictor_origin",
        "exif_transpose_applied",
    }
)

DEFAULT_QUALITY = 82


@dataclass(slots=True)
class EncodeOptions:
    """Normalised encoding knobs, independent of codec."""

    quality: int | None = None
    lossless: bool = False
    progressive: bool = False
    optimize: bool = True
    #: PNG zlib level 0-9, or None for the codec default.
    compress_level: int | None = None
    #: Codec effort where the format has one (WebP method 0-6, AVIF speed 0-10).
    effort: int | None = None
    subsampling: str | None = None
    #: Background used when flattening alpha for a codec without transparency.
    background: tuple[int, int, int] = (255, 255, 255)
    strip_metadata: bool = True
    #: Keep the ICC profile even when other metadata is stripped. Colour
    #: management is not privacy-sensitive, so this is often worth enabling.
    keep_icc: bool = False
    dpi: tuple[int, int] | None = None
    #: TIFF compression scheme.
    compression: str | None = None
    #: JPEG 2000 target compression ratios.
    quality_layers: list[float] | None = None
    irreversible: bool = True

    def normalised(self) -> EncodeOptions:
        """Clamp and validate the caller's numbers."""
        quality = self.quality
        if quality is not None and not 1 <= quality <= 100:
            raise InvalidArgumentError("quality must be between 1 and 100")
        compress_level = self.compress_level
        if compress_level is not None and not 0 <= compress_level <= 9:
            raise InvalidArgumentError("png_compress_level must be between 0 and 9")
        effort = self.effort
        if effort is not None and not 0 <= effort <= 10:
            raise InvalidArgumentError("effort must be between 0 and 10")
        red, green, blue = self.background
        for channel in (red, green, blue):
            if not 0 <= channel <= 255:
                raise InvalidArgumentError("background channels must be between 0 and 255")
        return self


@dataclass(slots=True)
class EncodedImage:
    """Encoded bytes plus what changed while producing them."""

    data: bytes
    spec: FormatSpec
    width: int
    height: int
    mode: str
    notes: list[str] = field(default_factory=list)

    @property
    def byte_size(self) -> int:
        return len(self.data)


def _has_alpha(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA", "PA", "RGBa", "La"}:
        return True
    return image.mode == "P" and "transparency" in image.info


def flatten_alpha(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """Composite an alpha image onto a solid background.

    Done explicitly so the caller's chosen background is honoured. Letting the
    codec decide produces black in some paths and white in others, which is a
    subtle way to corrupt a whole batch identically.
    """
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode == "LA":
        image = image.convert("RGBA")
    if image.mode != "RGBA":
        return image
    backdrop = Image.new("RGB", image.size, background)
    backdrop.paste(image, mask=image.getchannel("A"))
    return backdrop


def _coerce_mode(image: Image.Image, spec: FormatSpec, options: EncodeOptions) -> tuple[Image.Image, list[str]]:
    """Return an image whose mode ``spec`` can actually write."""
    notes: list[str] = []
    allowed = set(spec.write_modes)

    if image.mode in allowed:
        return image, notes

    if _has_alpha(image) and not spec.supports_alpha:
        notes.append(
            f"flattened transparency onto rgb{options.background} for {spec.label}, which has no alpha channel"
        )
        flattened = flatten_alpha(image, options.background)
        if flattened.mode in allowed:
            return flattened, notes
        image = flattened

    if image.mode in {"RGBA", "LA", "PA"} and "RGBA" in allowed:
        return image.convert("RGBA"), notes

    if image.mode in {"I", "I;16", "I;16B", "I;16L", "F"}:
        # High-bit-depth grayscale: scale into 8-bit rather than truncating.
        converted = image.convert("L")
        notes.append(f"downsampled {image.mode} to 8-bit grayscale")
        if converted.mode in allowed:
            return converted, notes
        image = converted

    if image.mode == "CMYK" and "CMYK" not in allowed:
        converted = image.convert("RGB")
        notes.append("converted CMYK to RGB")
        return converted, notes

    if image.mode == "P" and "P" not in allowed:
        converted = image.convert("RGBA" if "RGBA" in allowed else "RGB")
        return converted, notes

    if "RGB" in allowed:
        return image.convert("RGB"), notes
    if "RGBA" in allowed:
        return image.convert("RGBA"), notes
    if "P" in allowed:
        return _quantize(image), notes
    if "L" in allowed:
        return image.convert("L"), notes

    raise UnsupportedFormatError(
        f"cannot represent a {image.mode} image as {spec.label}",
        mode=image.mode,
        format=spec.key,
    )


def _quantize(image: Image.Image, colors: int = 256) -> Image.Image:
    """Convert to palette with an explicit, deterministic method."""
    if image.mode == "P":
        return image
    source = image.convert("RGBA") if _has_alpha(image) else image.convert("RGB")
    if source.mode == "RGBA":
        # Preserve a single transparent index where the format allows it.
        return source.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors)
    return source.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors)


def _jpeg_kwargs(options: EncodeOptions) -> dict[str, Any]:
    quality = options.quality or DEFAULT_QUALITY
    kwargs: dict[str, Any] = {
        "quality": quality,
        "optimize": options.optimize,
        "progressive": options.progressive,
    }
    if options.subsampling:
        kwargs["subsampling"] = options.subsampling
    elif quality >= 90:
        # At high quality the default 4:2:0 chroma subsampling is usually the
        # dominant artefact, so spend the bytes on full chroma instead.
        kwargs["subsampling"] = "4:4:4"
    return kwargs


def _png_kwargs(options: EncodeOptions) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"optimize": options.optimize}
    if options.compress_level is not None:
        kwargs["compress_level"] = options.compress_level
    return kwargs


def _webp_kwargs(options: EncodeOptions) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "quality": options.quality or DEFAULT_QUALITY,
        "lossless": options.lossless,
        "method": options.effort if options.effort is not None else 4,
    }
    return kwargs


def _avif_kwargs(options: EncodeOptions) -> dict[str, Any]:
    # The AVIF encoder has no `lossless` flag; libavif treats quality 100 as
    # (near-)lossless, so translate rather than silently ignoring the request.
    quality = 100 if options.lossless else (options.quality or DEFAULT_QUALITY)
    kwargs: dict[str, Any] = {"quality": quality}
    if options.effort is not None:
        # AVIF "speed" is inverted relative to WebP "method": lower is better.
        kwargs["speed"] = max(0, min(10, 10 - options.effort))
    if options.subsampling:
        kwargs["subsampling"] = options.subsampling
    return kwargs


def _tiff_kwargs(options: EncodeOptions) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if options.compression:
        kwargs["compression"] = options.compression
    elif options.lossless or options.quality is None:
        kwargs["compression"] = "tiff_deflate"
    else:
        kwargs["compression"] = "jpeg"
        kwargs["quality"] = options.quality
    return kwargs


def _jpeg2000_kwargs(options: EncodeOptions) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"irreversible": options.irreversible}
    if options.quality_layers:
        kwargs["quality_mode"] = "rates"
        kwargs["quality_layers"] = options.quality_layers
    return kwargs


def _gif_kwargs(options: EncodeOptions) -> dict[str, Any]:
    return {"optimize": options.optimize}


def _ico_kwargs(options: EncodeOptions) -> dict[str, Any]:
    return {}


_KWARG_BUILDERS = {
    "JPEG": _jpeg_kwargs,
    "PNG": _png_kwargs,
    "WEBP": _webp_kwargs,
    "AVIF": _avif_kwargs,
    "TIFF": _tiff_kwargs,
    "JPEG2000": _jpeg2000_kwargs,
    "GIF": _gif_kwargs,
    "ICO": _ico_kwargs,
    "BMP": lambda _o: {},
    "QOI": lambda _o: {},
    "PPM": lambda _o: {},
}

#: Optional keyword arguments per codec, in the order they may be dropped if the
#: encoder refuses them.
#:
#: Encoder builds differ enough that a keyword combination which works in one
#: libjpeg/libwebp build can fail in another - for example Pillow 12.3 raises
#: "broken data stream" for JPEG at ``subsampling="4:4:4"`` together with
#: ``optimize=True``. Rather than returning an error for a request that is
#: perfectly reasonable, the encoder retries without the least important option.
#:
#: Ordering encodes the tradeoff: for JPEG, chroma subsampling affects *image
#: quality* and is dropped last, while ``optimize`` only affects *file size* and
#: is dropped first.
_OPTIONAL_KWARGS: dict[str, tuple[str, ...]] = {
    "JPEG": ("progressive", "optimize", "subsampling"),
    "PNG": ("optimize", "compress_level"),
    "WEBP": ("method",),
    "AVIF": ("speed", "subsampling"),
    "TIFF": ("compression",),
    "JPEG2000": ("quality_layers", "irreversible"),
    "GIF": ("optimize",),
}


def _attempt_save(
    image: Image.Image,
    buffer: io.BytesIO,
    spec: FormatSpec,
    kwargs: dict[str, Any],
    guard: _MetadataGuard | None,
    notes: list[str],
    *,
    save_kwargs: dict[str, Any] | None = None,
) -> None:
    """Save, dropping optional keywords one at a time until the codec complies.

    Raises :class:`UnsupportedFormatError` only when the essential parameters
    also fail, so a genuine problem is still reported rather than masked.
    """
    attempt = {**(save_kwargs or {}), **kwargs}
    droppable = list(_OPTIONAL_KWARGS.get(spec.pillow_format, ()))

    while True:
        buffer.seek(0)
        buffer.truncate(0)
        try:
            if guard is not None:
                guard.__enter__()
            try:
                image.save(buffer, format=spec.pillow_format, **attempt)
            finally:
                if guard is not None:
                    guard.__exit__()
            return
        except (OSError, ValueError, KeyError) as exc:
            dropped = next((key for key in droppable if key in attempt), None)
            if dropped is None:
                raise UnsupportedFormatError(f"{spec.label} encoding failed: {exc}") from exc
            notes.append(f"{spec.label} encoder rejected '{dropped}' ({exc}); retried without it")
            logger.debug("dropping %s for %s after %s", dropped, spec.key, exc)
            attempt.pop(dropped)


def _metadata_kwargs(image: Image.Image, options: EncodeOptions) -> dict[str, Any]:
    """Explicit metadata arguments for the codecs that accept them."""
    kwargs: dict[str, Any] = {}
    if options.dpi:
        kwargs["dpi"] = options.dpi
    if options.strip_metadata and not options.keep_icc:
        return kwargs

    exif = image.info.get("exif")
    if exif and not options.strip_metadata:
        kwargs["exif"] = exif
    icc = image.info.get("icc_profile")
    if icc and (options.keep_icc or not options.strip_metadata):
        kwargs["icc_profile"] = icc
    return kwargs


class _MetadataGuard:
    """Temporarily hide metadata keys from ``image.info`` during a save.

    Pillow falls back to ``image.info`` for EXIF, ICC and PNG text chunks, so
    passing empty keyword arguments is not sufficient to drop them. Keys are
    removed for the duration of the save and restored afterwards, which avoids
    copying a large bitmap just to strip a few bytes.
    """

    __slots__ = ("_image", "_keys", "_saved")

    def __init__(self, image: Image.Image, keys: frozenset[str]) -> None:
        self._image = image
        self._keys = keys
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> None:
        info = self._image.info
        for key in self._keys:
            if key in info:
                self._saved[key] = info.pop(key)

    def __exit__(self, *_exc: object) -> None:
        self._image.info.update(self._saved)


def _square_for_icon(image: Image.Image) -> tuple[Image.Image, list[str]]:
    """Fit an image into a square canvas for ICO.

    ICO has no way to express a non-square image, and Pillow's encoder silently
    emits a cropped, squashed result (48x32 in becomes 32x21 out). Padding to a
    centred square on transparency is what a favicon consumer expects, and it
    keeps the whole image instead of trimming it.
    """
    if image.width == image.height:
        return image, []
    side = min(256, max(image.width, image.height))
    ratio = min(side / image.width, side / image.height)
    scaled_size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    scaled = image.convert("RGBA").resize(scaled_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.alpha_composite(scaled, ((side - scaled.width) // 2, (side - scaled.height) // 2))
    return canvas, [f"padded to a {side}x{side} square, as ICO requires"]


def encode(image: Image.Image, spec: FormatSpec, options: EncodeOptions | None = None) -> EncodedImage:
    """Encode ``image`` as ``spec``, returning the bytes and what changed."""
    opts = (options or EncodeOptions()).normalised()

    notes: list[str] = []
    if spec.pillow_format == "ICO":
        image, icon_notes = _square_for_icon(image)
        notes.extend(icon_notes)

    prepared, mode_notes = _coerce_mode(image, spec, opts)
    notes.extend(mode_notes)

    kwargs: dict[str, Any] = dict(_KWARG_BUILDERS.get(spec.pillow_format, lambda _o: {})(opts))
    kwargs.update(_metadata_kwargs(prepared, opts))
    if opts.lossless and spec.supports_lossless and spec.pillow_format in {"PNG", "WEBP"}:
        kwargs["lossless"] = True

    buffer = io.BytesIO()
    guard = _MetadataGuard(prepared, _METADATA_KEYS) if opts.strip_metadata else None
    _attempt_save(prepared, buffer, spec, kwargs, guard, notes)

    data = buffer.getvalue()
    if not data:
        raise UnsupportedFormatError(f"{spec.label} encoder produced no output")

    return EncodedImage(
        data=data,
        spec=spec,
        width=prepared.width,
        height=prepared.height,
        mode=prepared.mode,
        notes=notes,
    )


def encode_animation(
    frames: Sequence[Image.Image],
    spec: FormatSpec,
    options: EncodeOptions | None = None,
    *,
    durations: Sequence[int] | None = None,
    loop: int = 0,
) -> EncodedImage:
    """Encode a multi-frame image (GIF or WebP only).

    Frames are quantised against a shared palette so colours do not shimmer
    between frames, which is the usual artefact of per-frame adaptive palettes.
    """
    opts = (options or EncodeOptions()).normalised()
    if spec.pillow_format not in {"GIF", "WEBP", "AVIF", "TIFF"}:
        raise UnsupportedFormatError(
            f"{spec.label} does not support animation output",
            format=spec.key,
            animated_output=sorted({"gif", "webp", "avif", "tiff"}),
        )
    if not frames:
        raise InvalidArgumentError("no frames to encode")

    notes: list[str] = []
    prepared_frames = list(frames)

    if spec.pillow_format == "GIF":
        prepared_frames, quant_notes = _gif_frames(prepared_frames)
        notes.extend(quant_notes)
    else:
        converted: list[Image.Image] = []
        for frame in prepared_frames:
            coerced, frame_notes = _coerce_mode(frame, spec, opts)
            notes.extend(n for n in frame_notes if n not in notes)
            converted.append(coerced)
        prepared_frames = converted

    kwargs: dict[str, Any] = dict(_KWARG_BUILDERS.get(spec.pillow_format, lambda _o: {})(opts))
    kwargs.update(_metadata_kwargs(prepared_frames[0], opts))
    if opts.lossless and spec.pillow_format == "WEBP":
        kwargs["lossless"] = True

    first = prepared_frames[0]
    buffer = io.BytesIO()
    guard = _MetadataGuard(first, _METADATA_KEYS) if opts.strip_metadata else None
    _attempt_save(
        first,
        buffer,
        spec,
        kwargs,
        guard,
        notes,
        save_kwargs={
            "save_all": True,
            "append_images": prepared_frames[1:],
            "loop": loop,
            **({"duration": list(durations)} if durations else {}),
        },
    )

    data = buffer.getvalue()
    if not data:
        raise UnsupportedFormatError(f"{spec.label} encoder produced no output")

    return EncodedImage(
        data=data,
        spec=spec,
        width=first.width,
        height=first.height,
        mode=first.mode,
        notes=notes,
    )


def _gif_frames(frames: Sequence[Image.Image]) -> tuple[list[Image.Image], list[str]]:
    """Prepare frames for GIF without collapsing distinct frames.

    Deliberately *not* quantising here. An earlier version quantised every frame
    against the first frame's palette, which is wrong whenever the first frame
    is not representative: a palette built from a single flat colour maps every
    later frame to the same index, the frames become byte-identical, and Pillow
    then legitimately merges them into one - silently turning an animation into
    a still image.

    Pillow's GIF encoder already derives a palette that covers all frames when
    saving with ``save_all``, and it handles transparency, so the frames are
    passed through as RGBA and the quantisation is left to the codec.
    """
    prepared: list[Image.Image] = []
    converted_any = False
    for frame in frames:
        if frame.mode in {"P", "L"}:
            prepared.append(frame)
            continue
        prepared.append(frame.convert("RGBA") if _has_alpha(frame) else frame.convert("RGB"))
        converted_any = True
    notes = ["normalised frames for the GIF palette"] if converted_any else []
    return prepared, notes


__all__ = [
    "DEFAULT_QUALITY",
    "EncodeOptions",
    "EncodedImage",
    "encode",
    "encode_animation",
    "flatten_alpha",
]

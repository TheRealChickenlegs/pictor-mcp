"""Curated codec registry.

Pillow can open roughly forty formats. An image server should not, because each
decoder is a C library with its own history of memory-safety bugs, and most of
those formats have nothing to do with editing pictures on an internal network.

The registry is therefore an explicit allow-list:

* **Input formats** are limited to mainstream raster formats plus the open
  formats Pillow ships with (AVIF, JPEG 2000, QOI). PostScript, PDF, EPS, WMF
  and the scientific containers (HDF5, GRIB, FITS) are refused: they either
  shell out to external interpreters, are not images, or add a parser with no
  upside here.
* **Output formats** exclude PDF and EPS for the same reason - an "image
  converter" that will write a PDF is a document-forgery primitive.

The decoded ``Image.format`` is what decides, never the file extension and
never the caller's claim. A file called ``photo.png`` whose bytes are a
PostScript program is rejected at the point Pillow identifies it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from PIL import Image

from ..errors import UnsupportedFormatError


@dataclass(frozen=True, slots=True)
class FormatSpec:
    """Everything the server needs to know about one codec."""

    key: str
    pillow_format: str
    mime: str
    extension: str
    label: str
    supports_alpha: bool = False
    supports_animation: bool = False
    supports_quality: bool = False
    supports_lossless: bool = False
    supports_progressive: bool = False
    supports_metadata: bool = False
    supports_multipage: bool = False
    #: Modes this codec can write directly. Anything else needs conversion.
    write_modes: tuple[str, ...] = ("RGB",)
    notes: str = ""

    def to_public_dict(self) -> dict[str, object]:
        return {
            "format": self.key,
            "mime": self.mime,
            "extension": self.extension,
            "label": self.label,
            "supportsAlpha": self.supports_alpha,
            "supportsAnimation": self.supports_animation,
            "supportsQuality": self.supports_quality,
            "supportsLossless": self.supports_lossless,
            "supportsProgressive": self.supports_progressive,
            "supportsMetadata": self.supports_metadata,
        }


_SPECS: Final[tuple[FormatSpec, ...]] = (
    FormatSpec(
        key="jpeg",
        pillow_format="JPEG",
        mime="image/jpeg",
        extension=".jpg",
        label="JPEG",
        supports_quality=True,
        supports_progressive=True,
        supports_metadata=True,
        write_modes=("L", "RGB", "CMYK"),
        notes="Lossy, no transparency. The default for photographs.",
    ),
    FormatSpec(
        key="png",
        pillow_format="PNG",
        mime="image/png",
        extension=".png",
        label="PNG",
        supports_alpha=True,
        supports_lossless=True,
        supports_metadata=True,
        write_modes=("1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"),
        notes="Lossless, supports transparency. Best for graphics and screenshots.",
    ),
    FormatSpec(
        key="webp",
        pillow_format="WEBP",
        mime="image/webp",
        extension=".webp",
        label="WebP",
        supports_alpha=True,
        supports_animation=True,
        supports_quality=True,
        supports_lossless=True,
        supports_metadata=True,
        write_modes=("RGB", "RGBA"),
        notes="Modern web format with both lossy and lossless modes.",
    ),
    FormatSpec(
        key="avif",
        pillow_format="AVIF",
        mime="image/avif",
        extension=".avif",
        label="AVIF",
        supports_alpha=True,
        supports_animation=True,
        supports_quality=True,
        supports_lossless=True,
        supports_metadata=True,
        write_modes=("RGB", "RGBA"),
        notes="Best compression of the widely supported formats; slower to encode.",
    ),
    FormatSpec(
        key="tiff",
        pillow_format="TIFF",
        mime="image/tiff",
        extension=".tiff",
        label="TIFF",
        supports_alpha=True,
        supports_lossless=True,
        supports_metadata=True,
        supports_multipage=True,
        write_modes=("1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "I;16"),
        notes="Lossless archival format. Large files.",
    ),
    FormatSpec(
        key="gif",
        pillow_format="GIF",
        mime="image/gif",
        extension=".gif",
        label="GIF",
        supports_alpha=True,
        supports_animation=True,
        supports_lossless=True,
        write_modes=("P", "L", "RGB", "RGBA"),
        notes="256 colours only. Use for simple animation, not photographs.",
    ),
    FormatSpec(
        key="bmp",
        pillow_format="BMP",
        mime="image/bmp",
        extension=".bmp",
        label="BMP",
        supports_lossless=True,
        write_modes=("1", "L", "P", "RGB", "RGBA"),
        notes="Uncompressed. Rarely what you want.",
    ),
    FormatSpec(
        key="ico",
        pillow_format="ICO",
        mime="image/x-icon",
        extension=".ico",
        label="ICO",
        supports_alpha=True,
        supports_lossless=True,
        write_modes=("RGBA", "RGB", "P"),
        notes="Favicon container; supply square images.",
    ),
    FormatSpec(
        key="jpeg2000",
        pillow_format="JPEG2000",
        mime="image/jp2",
        extension=".jp2",
        label="JPEG 2000",
        supports_alpha=True,
        supports_quality=True,
        supports_lossless=True,
        write_modes=("L", "RGB", "RGBA", "I;16"),
        notes="Wavelet codec; excellent quality per byte, narrow support.",
    ),
    FormatSpec(
        key="qoi",
        pillow_format="QOI",
        mime="image/qoi",
        extension=".qoi",
        label="QOI",
        supports_alpha=True,
        supports_lossless=True,
        write_modes=("RGB", "RGBA"),
        notes="Trivially simple lossless codec.",
    ),
    FormatSpec(
        key="ppm",
        pillow_format="PPM",
        mime="image/x-portable-pixmap",
        extension=".ppm",
        label="Netpbm PPM",
        supports_lossless=True,
        write_modes=("1", "L", "RGB"),
        notes="Uncompressed interchange format.",
    ),
)

#: Canonical formats that may be written.
OUTPUT_FORMATS: Final[dict[str, FormatSpec]] = {spec.key: spec for spec in _SPECS}

#: Formats that may be decoded. Superset of outputs: PSD and MPO are readable
#: but not writable, and both are mainstream enough to accept.
_INPUT_ONLY: Final[dict[str, FormatSpec]] = {
    "psd": FormatSpec(
        key="psd",
        pillow_format="PSD",
        mime="image/vnd.adobe.photoshop",
        extension=".psd",
        label="Photoshop",
        supports_alpha=True,
        notes="Read-only; layers are flattened.",
    ),
    "mpo": FormatSpec(
        key="mpo",
        pillow_format="MPO",
        mime="image/jpeg",
        extension=".mpo",
        label="MPO (stereo JPEG)",
        supports_metadata=True,
        write_modes=("RGB",),
        notes="Read-only multi-picture JPEG.",
    ),
}

INPUT_FORMATS: Final[dict[str, FormatSpec]] = {**OUTPUT_FORMATS, **_INPUT_ONLY}

#: Pillow format string -> spec, for validating a decoded image.
_BY_PILLOW: Final[dict[str, FormatSpec]] = {spec.pillow_format: spec for spec in INPUT_FORMATS.values()}
_BY_MIME: Final[dict[str, FormatSpec]] = {spec.mime: spec for spec in INPUT_FORMATS.values()}
_BY_EXTENSION: Final[dict[str, FormatSpec]] = {}
for _spec in INPUT_FORMATS.values():
    _BY_EXTENSION[_spec.extension] = _spec
_BY_EXTENSION[".jpeg"] = OUTPUT_FORMATS["jpeg"]
_BY_EXTENSION[".jpe"] = OUTPUT_FORMATS["jpeg"]
_BY_EXTENSION[".jfif"] = OUTPUT_FORMATS["jpeg"]
_BY_EXTENSION[".tif"] = OUTPUT_FORMATS["tiff"]
_BY_EXTENSION[".jp2"] = OUTPUT_FORMATS["jpeg2000"]
_BY_EXTENSION[".j2k"] = OUTPUT_FORMATS["jpeg2000"]
_BY_EXTENSION[".webp"] = OUTPUT_FORMATS["webp"]

#: Friendly aliases accepted by tools.
_ALIASES: Final[dict[str, str]] = {
    "jpg": "jpeg",
    "jpe": "jpeg",
    "jfif": "jpeg",
    "tif": "tiff",
    "jp2": "jpeg2000",
    "j2k": "jpeg2000",
    "jpx": "jpeg2000",
    "netpbm": "ppm",
    "pgm": "ppm",
    "pbm": "ppm",
    "pnm": "ppm",
    "ico": "ico",
    "icon": "ico",
}


def _codec_available(pillow_format: str) -> bool:
    """True when Pillow can actually encode/decode this format here.

    AVIF and JPEG 2000 depend on build-time libraries, so a registry entry is a
    capability claim that must be checked rather than assumed.
    """
    Image.init()
    return pillow_format.upper() in Image.OPEN or pillow_format.upper() in Image.SAVE


def resolve_input_format(name: str) -> FormatSpec:
    """Resolve a caller-supplied format name to an accepted *input* codec."""
    spec = _lookup(name, INPUT_FORMATS)
    if spec is None:
        raise UnsupportedFormatError(
            f"unsupported image format {name!r}",
            supported=sorted(INPUT_FORMATS),
        )
    return spec


def resolve_output_format(name: str) -> FormatSpec:
    """Resolve a caller-supplied format name to a writable codec."""
    spec = _lookup(name, OUTPUT_FORMATS)
    if spec is None:
        raise UnsupportedFormatError(
            f"cannot write {name!r}",
            supported=sorted(OUTPUT_FORMATS),
        )
    if not _codec_available(spec.pillow_format):
        raise UnsupportedFormatError(
            f"this build of Pillow has no {spec.label} encoder",
            format=spec.key,
        )
    return spec


def _lookup(name: str, table: dict[str, FormatSpec]) -> FormatSpec | None:
    if not name:
        return None
    token = name.strip().lower().lstrip(".")
    token = _ALIASES.get(token, token)
    if token in table:
        return table[token]
    if token in _BY_MIME:
        spec = _BY_MIME[token]
        return spec if spec.key in table else None
    if "." + token in _BY_EXTENSION:
        spec = _BY_EXTENSION["." + token]
        return spec if spec.key in table else None
    return None


def spec_for_pillow_format(pillow_format: str | None) -> FormatSpec | None:
    if not pillow_format:
        return None
    return _BY_PILLOW.get(pillow_format.upper())


def assert_decoded_format_allowed(pillow_format: str | None, *, source: str = "input") -> FormatSpec:
    """Validate the format Pillow reported after opening the stream.

    This is the authoritative check: it does not trust the filename, the MIME
    type, or any caller assertion.
    """
    if not pillow_format:
        raise UnsupportedFormatError(
            f"{source} is not a recognised image (no decoder claimed it)",
        )
    spec = spec_for_pillow_format(pillow_format)
    if spec is None:
        raise UnsupportedFormatError(
            f"{source} is a {pillow_format} file, which this server does not decode",
            detected_format=pillow_format.lower(),
            supported=sorted(INPUT_FORMATS),
        )
    return spec


def public_format_catalogue() -> dict[str, list[dict[str, object]]]:
    """Format lists for the capabilities tool, filtered by real build support."""
    readable = [
        spec.to_public_dict()
        for spec in sorted(INPUT_FORMATS.values(), key=lambda s: s.key)
        if _codec_available(spec.pillow_format)
    ]
    writable = [
        spec.to_public_dict()
        for spec in sorted(OUTPUT_FORMATS.values(), key=lambda s: s.key)
        if _codec_available(spec.pillow_format)
    ]
    return {"readable": readable, "writable": writable}


__all__ = [
    "INPUT_FORMATS",
    "OUTPUT_FORMATS",
    "FormatSpec",
    "assert_decoded_format_allowed",
    "public_format_catalogue",
    "resolve_input_format",
    "resolve_output_format",
    "spec_for_pillow_format",
]

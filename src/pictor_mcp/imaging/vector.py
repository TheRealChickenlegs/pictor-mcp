"""SVG input support.

SVG is the one accepted input that is not a raster format, and it is deliberately
*not* handled by the shared Pillow decode path in
:mod:`pictor_mcp.imaging.loader`. Three properties make it different from every
other format this server reads:

* **Pillow cannot decode it.** There is no codec for it, so ``Image.open`` raises
  ``UnidentifiedImageError`` and the caller learns nothing about why - the generic
  "not a recognisable image" that this module exists to replace.
* **It is a document, not a bitmap.** Its raster size is a decision the server has
  to make - from ``width``/``height``, then ``viewBox``, then a default - and that
  decision is what bounds the work. A raster header, by contrast, only ever
  *reports* a size that :func:`~pictor_mcp.security.limits.check_decoded_geometry`
  can reject.
* **It is rendered by a native library that crashes the process on hostile
  input.** See :mod:`pictor_mcp.imaging._svgrender` for the two reproductions.

Accordingly this module does three things, in this order:

1. **Recognise** the document with :func:`looks_like_svg`, cheaply and without
   trusting the filename or ``media_type``.
2. **Inspect** it with :func:`inspect_svg`, which walks the document using
   ``expat`` callbacks *without building a tree* - so a deeply nested document is
   rejected rather than exhausting the stack - and reports the declared geometry.
3. **Rasterise** it with :func:`rasterize_svg`, which delegates to a child process
   and validates the returned bytes before they are handed to Pillow.

Nothing here trusts a declaration: the declared size is clamped against
``max_dimension`` and ``max_pixels`` before it is ever used as a render target.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import io
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.parsers import expat

from PIL import Image

from ..config import SvgPolicy
from ..errors import LimitExceededError, UnsupportedFormatError
from ._svgrender import EXIT_RENDERER_MISSING

logger = logging.getLogger(__name__)

#: The ``Image.format`` value that stands in for a rasterised SVG. Pillow itself
#: never reports this, so it is only ever produced by :func:`rasterize_svg`, and
#: registering it in the format allow-list is what lets the rasterised result flow
#: through the same validation as a real decode.
PILLOW_FORMAT = "SVG"

#: A cheap gate, not a verdict: bounded to the first few KiB and deliberately
#: forgiving. The authoritative check is :func:`inspect_svg`, which parses the
#: document. If this says "maybe" and the parse says "no", the caller falls back to
#: the ordinary Pillow path - so a JPEG whose EXIF comment happens to contain
#: ``<svg`` still decodes as a JPEG.
_PREFIX_BYTES = 8192
_ROOT_HINT = re.compile(rb"<(?:\s*[A-Za-z_][\w.\-]*:)?svg(?=[\s/>])", re.IGNORECASE)

#: Lengths are parsed from the SVG's own syntax. ``em``/``ex``/``rem``/``%`` are
#: viewport- or font-relative, so they are reported as "unknown" and resolved by
#: the default canvas instead of being guessed at.
_LENGTH = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z%]*)\s*$")
_UNIT_PX: dict[str, float] = {
    "": 1.0,
    "px": 1.0,
    "pt": 96.0 / 72.0,
    "pc": 16.0,
    "in": 96.0,
    "cm": 96.0 / 2.54,
    "mm": 96.0 / 25.4,
    "q": 96.0 / 101.6,
}

#: Slack allowed on top of the raw pixel buffer when bounding what the renderer
#: sends back. PNG framing costs more than this in no realistic case.
_PNG_OVERHEAD_ALLOWANCE = 8 * 1024 * 1024

#: Encodings tried for the document bytes. XML mandates that a document without a
#: BOM declare its encoding, and UTF-8 is the only encoding an XML parser must
#: support, so these two cover everything a caller legitimately produces.
_ENCODINGS = ("utf-8-sig", "utf-16")


class NotSvgDocumentError(Exception):
    """The bytes carry an ``<svg`` hint but are not an SVG document.

    A control-flow signal, never shown to a client: the caller falls back to the
    normal Pillow decode path, which is what keeps the prefix hint from turning a
    JPEG with an unlucky metadata string into a confusing SVG error.
    """


class _TooDeepError(Exception):
    """Internal: the document nests deeper than the configured ceiling."""


@dataclass(frozen=True, slots=True)
class SvgDocument:
    """What the server is willing to trust about an SVG before rendering it."""

    width: float | None
    height: float | None
    view_box: tuple[float, float, float, float] | None
    elements: int

    @property
    def declared_size(self) -> tuple[float, float] | None:
        """The intrinsic size in user units, or ``None`` when nothing declares one.

        A missing axis is derived from the other one through the ``viewBox`` aspect
        ratio, which is what a renderer would do; with no ``viewBox`` either, a
        single declared axis is treated as square.
        """
        box = self.view_box
        if self.width and self.height:
            return self.width, self.height
        if box is not None:
            box_width, box_height = box[2], box[3]
            if self.width:
                return self.width, self.width * box_height / box_width
            if self.height:
                return self.height * box_width / box_height, self.height
            return box_width, box_height
        if self.width:
            return self.width, self.width
        if self.height:
            return self.height, self.height
        return None


# --------------------------------------------------------------------- detection
def looks_like_svg(data: bytes) -> bool:
    """Whether ``data`` might be an SVG document.

    Only a bounded prefix is examined, and a match here is not a decision - see the
    module docstring. The point of keeping it cheap is that it runs on every
    decode, including for ordinary PNGs and JPEGs, where it costs one failed regex
    scan of the first few KiB.
    """
    if not data:
        return False
    return _ROOT_HINT.search(data[:_PREFIX_BYTES]) is not None


# ------------------------------------------------------------------- inspection
def _length_to_px(raw: str | None) -> float | None:
    """A declared SVG length in CSS pixels, or ``None`` if it is not absolute."""
    if not raw:
        return None
    match = _LENGTH.match(raw)
    if match is None:
        return None
    factor = _UNIT_PX.get(match.group(2).lower())
    if factor is None:
        return None
    try:
        value = float(match.group(1)) * factor
    except (ValueError, OverflowError):
        return None
    # "1e999" parses to inf; a non-finite length is not a size that can be rendered.
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _parse_view_box(raw: str | None) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    parts = raw.replace(",", " ").split()
    if len(parts) != 4:
        return None
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    if values[2] <= 0 or values[3] <= 0:
        return None
    return values  # type: ignore[return-value]


def _local_name(name: str) -> str:
    """``svg:svg`` -> ``svg``, so a prefixed root is still recognised."""
    return name.rsplit(":", 1)[-1]


def inspect_svg(data: bytes, *, max_depth: int) -> SvgDocument:
    """Parse the document's structure, refusing anything unreasonable.

    The document is walked with ``expat`` callbacks and no tree is materialised, so
    nesting depth costs one integer instead of one Python frame per level - a
    document nested 100000 deep is *rejected*, not a ``RecursionError``.

    Entity declarations are deliberately tolerated: real-world SVG from Illustrator
    and Inkscape carries a ``DOCTYPE`` whose internal subset declares unused
    namespace entities, and the renderer ignores them. ``expat`` does not resolve
    external entities, so the ``SYSTEM``/``PUBLIC`` identifiers in such a
    ``DOCTYPE`` are never fetched.

    Raises :class:`NotSvgDocumentError` when the first element is not ``svg``, which
    is the signal to fall back to the Pillow path.
    """
    root_attrs: dict[str, str] | None = None
    first_element: str | None = None
    depth = 0
    elements = 0

    def start_element(name: str, attrs: dict[str, str]) -> None:
        nonlocal root_attrs, first_element, depth, elements
        depth += 1
        elements += 1
        if depth > max_depth:
            raise _TooDeepError
        if depth == 1:
            first_element = name
            root_attrs = attrs

    def end_element(_name: str) -> None:
        nonlocal depth
        depth -= 1

    parser = expat.ParserCreate()
    parser.StartElementHandler = start_element
    parser.EndElementHandler = end_element
    # Text is never inspected, so it is never accumulated.
    parser.buffer_text = False

    try:
        parser.Parse(data, True)
    except _TooDeepError as exc:
        raise LimitExceededError(
            f"SVG nests more than {max_depth} levels deep, which is never a real document",
            limit=max_depth,
        ) from exc
    except expat.ExpatError as exc:
        if first_element is None or _local_name(first_element).lower() != "svg":
            # Nothing that could be called an SVG was ever seen: the prefix hint was
            # a false positive, so let Pillow have the bytes.
            raise NotSvgDocumentError from exc
        raise UnsupportedFormatError(
            "SVG document could not be parsed as XML",
            line=exc.lineno,
            column=exc.offset,
        ) from exc

    if root_attrs is None or first_element is None or _local_name(first_element).lower() != "svg":
        raise NotSvgDocumentError("the document's root element is not 'svg'")

    return SvgDocument(
        width=_length_to_px(root_attrs.get("width")),
        height=_length_to_px(root_attrs.get("height")),
        view_box=_parse_view_box(root_attrs.get("viewBox")),
        elements=elements,
    )


def render_target(
    doc: SvgDocument,
    policy: SvgPolicy,
    *,
    max_dimension: int,
    max_pixels: int,
) -> tuple[int, int, bool]:
    """Decide the pixel size to render at, within the configured ceilings.

    Returns ``(width, height, was_reduced)``. The size is clamped *before* the
    renderer is started, which is the whole defence against a document that
    declares an enormous canvas: the work is bounded by this decision, never by the
    document's own claim.
    """
    declared = doc.declared_size
    width, height = declared if declared is not None else (float(policy.default_size),) * 2

    area = width * height
    if not (math.isfinite(width) and math.isfinite(height) and math.isfinite(area)) or width <= 0 or height <= 0:
        # An infinite or nonsensical declaration is not a canvas; use the default.
        width = height = float(policy.default_size)
        area = width * height

    scale = min(1.0, max_dimension / width, max_dimension / height, math.sqrt(max_pixels / area))
    reduced = scale < 1.0
    target_width = max(1, min(max_dimension, int(width * scale)))
    target_height = max(1, min(max_dimension, int(height * scale)))
    return target_width, target_height, reduced


# ---------------------------------------------------------------------- renderer
@lru_cache(maxsize=1)
def renderer_version() -> str | None:
    """The installed ``resvg-py`` version, or ``None`` when it is absent.

    Deliberately does not import ``resvg_py``. Importing it would load the native
    library into the long-lived server process, which is the one thing the
    subprocess boundary exists to prevent, so availability is answered from the
    import system and the package metadata alone - neither of which executes the
    renderer.
    """
    try:
        found = importlib.util.find_spec("resvg_py") is not None
    except (ImportError, ValueError):
        # A broken parent package raises rather than returning None; that counts as
        # "not usable here" just as much as a missing one does.
        logger.debug("the SVG renderer spec could not be resolved", exc_info=True)
        return None
    if not found:
        return None
    try:
        return importlib.metadata.version("resvg-py")
    except importlib.metadata.PackageNotFoundError:
        # Present but installed outside the metadata database, e.g. vendored.
        return "unknown"


def renderer_available() -> bool:
    """Whether SVG input can be rasterised at all in this installation."""
    return renderer_version() is not None


def renderer_label() -> str:
    version = renderer_version()
    return f"resvg {version}" if version else "unavailable"


def _child_command(policy: SvgPolicy, width: int, height: int) -> list[str]:
    # -B keeps the child from writing __pycache__ into the package during a
    # hostile-input render (and from warning on a read-only root filesystem).
    command = [
        sys.executable,
        "-B",
        "-m",
        "pictor_mcp.imaging._svgrender",
        "--width",
        str(width),
        "--height",
        str(height),
        "--memory-limit-mib",
        str(policy.memory_limit_mib),
        "--cpu-limit-seconds",
        str(math.ceil(policy.timeout_seconds) + 5),
    ]
    if not policy.system_fonts:
        command.append("--no-system-fonts")
    return command


def _child_environment() -> dict[str, str]:
    """The child's environment: enough to import this package, and nothing else.

    Every ``PICTOR_*`` variable is stripped, because the child needs none of them -
    the render parameters arrive as arguments - and two of them are secrets
    (``PICTOR_AUTH_TOKEN`` and ``PICTOR_URL_SECRET``). The child parses hostile
    input with a native library, so it is the last place those should be readable
    from.

    The directory containing the package is prepended to ``PYTHONPATH`` so the child
    works when the server runs from a source checkout that was never pip-installed -
    the same arrangement ``tests/conftest.py`` relies on.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("PICTOR_")}
    package_parent = str(Path(__file__).resolve().parents[2])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(part for part in (package_parent, existing) if part)
    # Neither may influence the child: one runs an arbitrary file at interpreter
    # start-up, the other switches the interpreter into an interactive mode.
    env.pop("PYTHONSTARTUP", None)
    env.pop("PYTHONINSPECT", None)
    return env


def rasterize_svg(
    data: bytes,
    policy: SvgPolicy,
    *,
    max_dimension: int,
    max_pixels: int,
) -> tuple[Image.Image, list[str]]:
    """Render ``data`` to a Pillow image in an isolated child process.

    Raises a client-safe :class:`~pictor_mcp.errors.PictorError` for every failure
    mode, including the child being killed by a signal - which is the expected
    outcome for the documents described in :mod:`pictor_mcp.imaging._svgrender`.
    """
    # Parse first: only a document whose root element really is <svg> should ever be
    # reported as an SVG problem. Everything else raises NotSvgDocumentError and the
    # caller falls back to Pillow.
    doc = inspect_svg(data, max_depth=policy.max_depth)

    if not policy.enabled:
        raise UnsupportedFormatError(
            "SVG input is disabled; set PICTOR_ALLOW_SVG=true to rasterise vector files",
            detected_format="svg",
        )
    if not renderer_available():
        raise UnsupportedFormatError(
            "SVG input needs the 'resvg-py' renderer, which is not installed in this environment",
            detected_format="svg",
        )

    width, height, reduced = render_target(doc, policy, max_dimension=max_dimension, max_pixels=max_pixels)

    notes = [f"SVG rasterised to {width}x{height} by {renderer_label()} in an isolated renderer process"]
    if reduced:
        notes.append(
            f"the declared size was reduced to {width}x{height} to stay within the "
            f"{max_dimension}px per-axis and {max_pixels} pixel limits"
        )

    png = _run_renderer(data, policy, width=width, height=height)

    try:
        # Pillow is never handed a path, only the PNG bytes just validated above. The
        # stream is dropped after load(), the same contract loader.decode_output_bytes
        # relies on.
        image = Image.open(io.BytesIO(png))
        image.load()
    except Exception as exc:
        raise UnsupportedFormatError("the SVG renderer did not produce a valid image") from exc

    return image, notes


def _run_renderer(data: bytes, policy: SvgPolicy, *, width: int, height: int) -> bytes:
    """Run the child and return the PNG it produced.

    ``subprocess.run`` is used rather than a hand-rolled reader because it handles
    the three deadlock-prone cases at once: a large stdin write, concurrent stdout
    and stderr, and killing the child on timeout. The price is that stdout is
    buffered before it can be measured, so the size bound below is applied after the
    fact rather than enforced during the read. That is acceptable because the target
    size - and therefore the renderer's legitimate output - is something this server
    chose a few lines earlier, and it is capped by the same pixel ceilings that apply
    to any other decode.
    """
    limit = _png_size_bound(width, height)
    with tempfile.TemporaryFile() as errors:
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, never a shell
                _child_command(policy, width, height),
                input=data,
                stdout=subprocess.PIPE,
                stderr=errors,
                timeout=policy.timeout_seconds,
                env=_child_environment(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise LimitExceededError(
                f"SVG rendering exceeded the {policy.timeout_seconds:g} second limit",
                limit=policy.timeout_seconds,
            ) from exc
        except OSError as exc:
            raise UnsupportedFormatError("the SVG renderer process could not be started") from exc

        detail = _read_diagnostics(errors)

    if completed.returncode != 0:
        # A negative status means the child died on that signal - the documented
        # outcome for a document that overflows the renderer's stack. Either way the
        # server is unharmed, which is the entire reason for the subprocess.
        logger.warning("SVG renderer exited with status %s: %s", completed.returncode, detail)
        if completed.returncode == EXIT_RENDERER_MISSING:
            # A distinct code, because "install the dependency" and "this document is
            # unacceptable" are different problems with different fixes.
            raise UnsupportedFormatError(
                "SVG input needs the 'resvg-py' renderer, which is installed but could not be loaded",
                detected_format="svg",
            )
        raise UnsupportedFormatError(
            "this SVG could not be rasterised; the renderer rejected or crashed on it",
            detected_format="svg",
        )

    png = completed.stdout or b""
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        logger.warning("SVG renderer produced non-PNG output: %s", detail)
        raise UnsupportedFormatError("the SVG renderer did not produce a PNG image")
    if len(png) > limit:
        raise LimitExceededError(
            f"the rendered SVG is {len(png)} bytes, exceeding the {limit} byte bound for a {width}x{height} image",
            limit=limit,
        )
    return png


def _png_size_bound(width: int, height: int) -> int:
    """Upper bound on the encoded size accepted for a render of this size.

    Uncompressed RGBA plus PNG framing is the worst case for a single frame, so
    anything larger is a misbehaving renderer rather than a legitimate image.
    """
    return width * height * 4 + _PNG_OVERHEAD_ALLOWANCE


def _read_diagnostics(handle: Any) -> str:
    """Collect the child's diagnostics for the server log, bounded."""
    try:
        handle.seek(0)
        raw = handle.read(4096)
    except Exception:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace").strip()
    return str(raw).strip()


__all__ = [
    "PILLOW_FORMAT",
    "NotSvgDocumentError",
    "SvgDocument",
    "inspect_svg",
    "looks_like_svg",
    "rasterize_svg",
    "render_target",
    "renderer_available",
    "renderer_label",
    "renderer_version",
]

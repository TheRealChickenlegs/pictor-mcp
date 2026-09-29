"""Isolated SVG rasteriser, run as a *child process* and never imported.

:mod:`pictor_mcp.imaging.vector` launches this module with

.. code-block:: text

    python -B -m pictor_mcp.imaging._svgrender --width W --height H

writes the SVG document to stdin and expects PNG bytes on stdout. The protocol is
deliberately one-way and argument-free beyond integer dimensions: the child never
receives a path, a URL or a filename, so nothing in the document can influence
what it opens.

Why a subprocess
----------------
The renderer is a native library that *aborts the process* on hostile input, and
no input filter can be relied on to prevent that. Two reproductions on
``resvg-py`` 0.5.0 (macOS arm64, CPython 3.12):

* a chain of 300 sibling ``<pattern>`` elements - about 20 KiB of XML at depth 5,
  with no unusual constructs - raises SIGSEGV deterministically;
* ``<g>`` nesting around 1000 deep raises SIGSEGV.

Neither is a memory-exhaustion attack that a pixel ceiling would stop; both are
stack overflows inside the renderer. A segfault cannot be caught from Python, so
in-process rendering would mean one crafted 20 KiB file kills the whole server for
every client. Bounding the input instead is not sound: the failing depth depends
on the stack the renderer happens to run with (a worker thread, and far smaller
again under musl/Alpine), so a limit tuned on one machine is not a limit on
another.

The child is therefore treated as expendable. If it dies, the parent observes a
negative exit status and reports a failed tool call. The costs of the extra
process are one interpreter start plus a Rust extension import, measured at about
18 ms per render on the machine above.

The child is also where the resource ceilings are applied - ``RLIMIT_AS`` and
``RLIMIT_CPU`` - rather than through ``preexec_fn`` in the parent, because the
parent is multi-threaded and ``preexec_fn`` is documented as unsafe there.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys

#: Exit codes, so the parent can tell a refused document from a crash, and a crash
#: from a renderer that is not installed at all.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BAD_INPUT = 3
EXIT_RENDER_FAILED = 4
EXIT_RENDERER_MISSING = 5

#: Encodings tried, in order, for the document bytes. XML mandates that a
#: document without a BOM declare its encoding, and UTF-8 is the only encoding
#: an XML parser must support, so these two cover everything a caller legitimately
#: produces.
_ENCODINGS = ("utf-8-sig", "utf-16")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pictor-mcp-svgrender", add_help=False)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--no-system-fonts", action="store_true")
    parser.add_argument("--memory-limit-mib", type=int, default=0)
    parser.add_argument("--cpu-limit-seconds", type=float, default=0.0)
    return parser.parse_args(argv)


def _apply_limits(memory_limit_mib: int, cpu_limit_seconds: float) -> None:
    """Cap address space and CPU time for the render, where the OS supports it.

    Applied here rather than in the parent's ``preexec_fn``: the parent runs this
    from a worker thread, and ``preexec_fn`` is explicitly documented as unsafe in
    a threaded program. Limits are a backstop only - the parent's own wall-clock
    timeout is what actually bounds a hang.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows has no resource module
        return

    def _set(which: int, soft: int, hard: int) -> None:
        with contextlib.suppress(ValueError, OSError):  # pragma: no cover - not enforceible here
            resource.setrlimit(which, (soft, hard))

    if memory_limit_mib > 0:
        limit = memory_limit_mib * 1024 * 1024
        _set(resource.RLIMIT_AS, limit, limit)
    if cpu_limit_seconds > 0:
        soft = math.ceil(cpu_limit_seconds)
        _set(resource.RLIMIT_CPU, soft, soft + 2)


def _decode_document(data: bytes) -> str:
    for encoding in _ENCODINGS:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("the SVG document is not valid UTF-8 or UTF-16 text")


def _load_renderer():
    """Import the renderer, or fail with a nameable exit code.

    Returned rather than imported at module scope so that a missing or unloadable
    native library produces ``EXIT_RENDERER_MISSING`` instead of a traceback and a
    bare exit 1 - the parent turns that one code into a message naming the package
    to install.
    """
    try:
        import resvg_py
    except Exception as exc:
        print(f"resvg_py could not be imported: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    return resvg_py


def _render(renderer: object, text: str, width: int, height: int, *, system_fonts: bool) -> bytes:
    # No resources_dir and no font files are passed: the only filesystem access the
    # renderer performs is reading the system font directories, and that is disabled
    # entirely by --no-system-fonts. External references in the document (file:,
    # http:, https:) are therefore never resolved - verified by probing that an
    # <image href="file:///etc/passwd"> renders as an empty canvas.
    return renderer.svg_to_bytes(  # type: ignore[attr-defined]
        svg_string=text,
        width=width,
        height=height,
        skip_system_fonts=not system_fonts,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        print(f"render size must be positive (got {args.width}x{args.height})", file=sys.stderr)
        return EXIT_USAGE

    payload = sys.stdin.buffer.read()
    try:
        text = _decode_document(payload)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_BAD_INPUT

    renderer = _load_renderer()
    if renderer is None:
        return EXIT_RENDERER_MISSING

    _apply_limits(args.memory_limit_mib, args.cpu_limit_seconds)

    try:
        png = _render(renderer, text, args.width, args.height, system_fonts=not args.no_system_fonts)
    except Exception as exc:
        # A library-internal string, deliberately written to the child's stderr - which
        # the parent logs but never returns to a client - and never to stdout.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RENDER_FAILED

    if not png:
        print("renderer produced no output", file=sys.stderr)
        return EXIT_RENDER_FAILED

    sys.stdout.buffer.write(png)
    sys.stdout.buffer.flush()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

"""MCP resources exposing previously generated files.

A ``resource_link`` in a tool result points at ``pictor://outputs/<path>``.
Clients that understand resources can fetch the bytes directly instead of
asking a tool to re-encode them, which is both cheaper and avoids a second
lossy pass.

Reads are confined to the output root by the same jail used for writes, and the
SDK's resource security additionally rejects traversal, absolute paths and NUL
bytes in the template parameter. The two checks are independent on purpose.
"""

from __future__ import annotations

import logging
import mimetypes
from functools import partial
from typing import TYPE_CHECKING

import anyio

from .errors import PathNotAllowedError
from .outputs import RESOURCE_PREFIX

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer

    from .tools.context import ToolContext

logger = logging.getLogger(__name__)

#: Refuse to serve anything above this size through a resource read.
_MAX_RESOURCE_BYTES = 256 * 1024 * 1024


def register_resources(server: MCPServer, ctx: ToolContext) -> None:
    """Register the output-file resource template."""

    @server.resource(
        f"pictor://{RESOURCE_PREFIX}/{{+path}}",
        name="Generated image",
        title="Generated image",
        description="Bytes of a file this server wrote to its output root.",
        mime_type="application/octet-stream",
    )
    async def read_generated_image(path: str) -> bytes:
        try:
            return await anyio.to_thread.run_sync(
                partial(ctx.jail.read_output_bytes, path, max_bytes=_MAX_RESOURCE_BYTES)
            )
        except PathNotAllowedError as exc:
            logger.info("resource read rejected for %r", path)
            raise PathNotAllowedError("resource is not available") from exc


#: Explicit extension -> MIME map for everything this server can write.
#:
#: ``mimetypes.guess_type`` reads the host's ``/etc/mime.types``, which is
#: missing from many slim container images - ``.webp`` and ``.avif`` in
#: particular come back as ``application/octet-stream`` there, and a browser
#: will not render an image served with the wrong content type. The container's
#: mimetypes database is therefore not something this server should depend on.
_IMAGE_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jpe": "image/jpeg",
    ".jfif": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".ico": "image/x-icon",
    ".jp2": "image/jp2",
    ".j2k": "image/jp2",
    ".qoi": "image/qoi",
    ".ppm": "image/x-portable-pixmap",
    ".pgm": "image/x-portable-graymap",
    ".pbm": "image/x-portable-bitmap",
}


#: Content types the signed file endpoint is willing to serve *as that type*.
#:
#: Everything else is served as ``application/octet-stream`` with
#: ``Content-Disposition: attachment``. A generated file must never be returned
#: as ``text/html``: metadata (an ICC profile, an EXIF comment) or a caller's
#: chosen filename could otherwise put attacker-controlled markup on the
#: server's own origin, which is stored XSS. ``nosniff`` does not help when the
#: server states the type explicitly, and the extension is attacker-influenced.
_SERVABLE_MIME: frozenset[str] = frozenset(_IMAGE_MIME.values())


def guess_mime(path: str) -> str:
    """Content type for a generated file, without trusting the host database."""
    lowered = path.lower()
    for extension, mime in _IMAGE_MIME.items():
        if lowered.endswith(extension):
            return mime
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def serve_mime(path: str) -> tuple[str, bool]:
    """Return ``(content_type, inline)`` for a served file.

    ``inline`` is true only for a recognised image type. Anything else - a name
    with an unexpected extension, or a type the image allow-list does not cover -
    is deliberately downgraded to an opaque download.
    """
    content_type = guess_mime(path)
    if content_type in _SERVABLE_MIME:
        return content_type, True
    return "application/octet-stream", False


__all__ = ["guess_mime", "register_resources", "serve_mime"]

"""Output storage and result assembly.

Every tool writes its result to the output root *and* describes it in the
response. Writing to disk is what makes the server usable from a terminal, a
compose volume, a reverse proxy, or a web UI; the response envelope is what
makes it usable from a model. Neither alone is sufficient.

The envelope carries the same artefact in up to four forms, and each is
optional so no client is forced to handle a shape it cannot render:

============ ==========================================================
``text``     Always present. Self-contained: sizes, dimensions, savings,
             the relative path and any URL, so a text-only client loses
             nothing but the pixels.
``image``    Present when inlining is on and the payload is under the
             inline cap. Vision-capable clients render it directly.
``link``     A ``resource_link`` for clients that resolve MCP resources.
``structured`` Full JSON for programmatic callers.
============ ==========================================================

Base64 is never duplicated between the image block and the JSON: doubling a
payload in a model's context is a real cost and a common oversight.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from mcp.server.mcpserver import Image as MCPImage
from mcp_types import CallToolResult, ResourceLink, TextContent
from pydantic import BaseModel

from .config import Config
from .errors import InvalidArgumentError
from .imaging.encode import EncodedImage
from .imaging.formats import FormatSpec
from .imaging.loader import LoadedImage, display_name
from .imaging.ops import Size
from .models import FileOutput, ImageResult, SizeChange
from .security.auth import build_signed_path
from .security.paths import PathJail, safe_filename

logger = logging.getLogger(__name__)

#: URI scheme for artefact resources.
RESOURCE_SCHEME = "pictor"
RESOURCE_PREFIX = "outputs"

_MAX_UNIQUE_ATTEMPTS = 200


def human_bytes(value: int | None) -> str:
    """Format a byte count compactly."""
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


@dataclass(slots=True)
class StoredOutput:
    """An encoded artefact written to the output root."""

    file: FileOutput
    data: bytes
    absolute_path: Path


class ResultBuilder:
    """Writes artefacts and assembles MCP tool results."""

    __slots__ = ("_config", "_jail")

    def __init__(self, config: Config, jail: PathJail) -> None:
        self._config = config
        self._jail = jail

    # ------------------------------------------------------------- storage
    def store(
        self,
        encoded: EncodedImage,
        *,
        subdirectory: str,
        filename: str,
        include_base64: bool = False,
    ) -> StoredOutput:
        """Write encoded bytes under ``output_root/subdirectory``."""
        if not subdirectory or "/" in subdirectory.strip("/") or ".." in subdirectory:
            raise InvalidArgumentError("output subdirectory must be a single safe path segment")

        directory = self._config.output_root / subdirectory
        target = self._unique_target(directory, filename)
        written = self._jail.write_bytes(target, encoded.data, overwrite=False)

        relative = self._jail.relative_output(written)
        digest = hashlib.sha256(encoded.data).hexdigest()

        payload = encoded.data if include_base64 else None
        file_output = FileOutput(
            name=written.name,
            path=relative,
            mimeType=encoded.spec.mime,
            format=encoded.spec.key,
            byteSize=len(encoded.data),
            width=encoded.width,
            height=encoded.height,
            sha256=digest,
            url=self._public_url(relative),
            base64=base64.b64encode(payload).decode() if payload is not None else None,
        )
        return StoredOutput(file=file_output, data=encoded.data, absolute_path=written)

    def _unique_target(self, directory: Path, filename: str) -> Path:
        """Return a non-colliding path, keeping the readable name where possible."""
        directory.mkdir(parents=True, exist_ok=True, mode=0o750)
        candidate = directory / filename
        if not candidate.exists():
            return candidate

        stem = Path(filename).stem
        suffix = Path(filename).suffix
        for counter in range(1, _MAX_UNIQUE_ATTEMPTS):
            candidate = directory / f"{stem}-{counter}{suffix}"
            if not candidate.exists():
                return candidate
        # Fall back to a random token rather than overwriting anything.
        token = hashlib.sha256(str(directory / filename).encode()).hexdigest()[:10]
        return directory / f"{stem}-{token}{suffix}"

    def _public_url(self, relative: str) -> str | None:
        http = self._config.http
        if not http.serve_outputs:
            return None
        secret = http.url_secret
        signed = build_signed_path(secret, relative, http.url_ttl_seconds) if secret else relative
        base = http.public_base_url
        if not base:
            # Comparing against the wildcard bind address, not binding to it.
            wildcard = {"0.0.0.0", "::"}  # noqa: S104
            host = "127.0.0.1" if http.host in wildcard else http.host
            base = f"http://{host}:{http.port}"
        return f"{base}/files/{signed}"

    # ------------------------------------------------------------- results
    def build_image_result(
        self,
        *,
        operation: str,
        outputs: list[StoredOutput],
        input_summary: dict | None = None,
        input_bytes: int | None = None,
        notes: list[str] | None = None,
        steps: list[str] | None = None,
        metrics: dict | None = None,
        inline: bool = False,
    ) -> CallToolResult:
        """Assemble a standard image result."""
        collected_notes = list(notes or [])
        primary = outputs[0] if outputs else None
        total_output = sum(item.file.byte_size for item in outputs) or None

        inline_bytes: bytes | None = None
        inline_mime: str | None = None
        wants_inline = inline and self._config.inline_images and primary is not None
        if wants_inline and primary is not None:
            limit = self._config.limits.max_inline_image_bytes
            if len(primary.data) <= limit:
                inline_bytes = primary.data
                inline_mime = primary.file.mime_type
            else:
                collected_notes.append(
                    f"image not embedded inline: {human_bytes(len(primary.data))} exceeds the "
                    f"{human_bytes(limit)} inline limit"
                )

        result = ImageResult(
            operation=operation,
            outputs=[item.file for item in outputs],
            input=input_summary,
            sizeChange=SizeChange(
                inputBytes=input_bytes,
                outputBytes=total_output,
                savedBytes=(input_bytes - total_output) if input_bytes and total_output else None,
                savedPercent=(
                    round((1 - total_output / input_bytes) * 100, 2) if input_bytes and total_output else None
                ),
            )
            if input_bytes or total_output
            else None,
            notes=collected_notes,
            steps=steps or [],
            metrics=metrics,
            inlineImageIncluded=inline_bytes is not None,
        )
        return self._assemble(result, inline_bytes=inline_bytes, inline_mime=inline_mime)

    def build_model_result(
        self,
        result: ImageResult,
        *,
        inline_bytes: bytes | None = None,
        inline_mime: str | None = None,
    ) -> CallToolResult:
        """Assemble a result from an already-built model."""
        return self._assemble(result, inline_bytes=inline_bytes, inline_mime=inline_mime)

    def _assemble(
        self,
        result: ImageResult,
        *,
        inline_bytes: bytes | None,
        inline_mime: str | None,
    ) -> CallToolResult:
        links = [
            ResourceLink(
                type="resource_link",
                uri=f"{RESOURCE_SCHEME}://{RESOURCE_PREFIX}/{output.path}",
                name=output.name,
                mimeType=output.mime_type,
                description=f"{output.format} {output.width}x{output.height}",
            )
            for output in result.outputs
        ]
        return build_result(
            result,
            self.summarise(result),
            inline_bytes=inline_bytes,
            inline_mime=inline_mime,
            links=links,
        )

    # ------------------------------------------------------------- summary
    def summarise(self, result: ImageResult) -> str:
        """Render a self-contained text summary."""
        lines: list[str] = []
        headline = result.operation.replace("_", " ").strip().capitalize()
        lines.append(f"{headline} completed." if result.ok else f"{headline} failed.")

        if result.input:
            lines.append(
                "Input: {w}x{h} {fmt} {size}".format(
                    w=result.input.get("width", "?"),
                    h=result.input.get("height", "?"),
                    fmt=str(result.input.get("format", "?")).upper(),
                    size=human_bytes(result.input.get("byteSize")),
                )
            )

        for output in result.outputs:
            lines.append(
                f"Output: {output.width}x{output.height} {output.format.upper()} {human_bytes(output.byte_size)} -> {output.path}"
            )
            if output.url:
                lines.append(f"Image URL: {output.url}")

        change = result.size_change
        if change and change.input_bytes and change.output_bytes:
            percent = change.saved_percent or 0.0
            direction = "smaller" if percent >= 0 else "larger"
            lines.append(
                f"Size: {human_bytes(change.input_bytes)} -> {human_bytes(change.output_bytes)} "
                f"({abs(percent):.1f}% {direction})"
            )

        if result.metrics:
            rendered = ", ".join(f"{key}={value}" for key, value in result.metrics.items())
            lines.append(f"Metrics: {rendered}")

        if result.steps:
            lines.append(f"Steps: {' -> '.join(result.steps)}")

        for note in result.notes:
            lines.append(f"Note: {note}")

        # Last, deliberately. Chat UIs render markdown from the assistant's
        # reply, not from a tool result, and Open WebUI does not display the MCP
        # image block at all (open-webui discussion #14732). A URL on its own
        # therefore disappears: the model summarises the run and the picture
        # never appears. Spelling out the exact line to send is what makes it
        # show up, and putting it last is what makes a model repeat it.
        #
        # The alt text is the literal word `image`, not the file name: it is
        # `![image](url)` that clients and models reproduce verbatim, and a name
        # with dots, dashes or spaces is one more thing to retype slightly
        # differently - which is the whole failure mode being worked around.
        linked = [output for output in result.outputs if output.url]
        if linked:
            lines.append("")
            lines.append("To display the result, copy this into your reply exactly as written:")
            lines.extend(f"![image]({output.url})" for output in linked)

        return "\n".join(lines)


def build_result(
    model: BaseModel,
    text: str,
    *,
    inline_bytes: bytes | None = None,
    inline_mime: str | None = None,
    links: list[ResourceLink] | None = None,
) -> CallToolResult:
    """Assemble a ``CallToolResult`` carrying text, optional image, links and JSON.

    The text block always comes first so a client that renders only the first
    content block still shows something useful.
    """
    content: list = [TextContent(type="text", text=text)]

    if inline_bytes is not None and inline_mime is not None:
        try:
            content.append(MCPImage(data=inline_bytes, format=inline_mime.split("/", 1)[-1]).to_image_content())
        except Exception:  # pragma: no cover - defensive
            logger.warning("failed to embed inline image", exc_info=True)

    if links:
        content.extend(links)

    return CallToolResult(
        content=content,
        structuredContent=model.model_dump(mode="json", by_alias=True, exclude_none=True),
    )


def default_filename(loaded: LoadedImage, spec: FormatSpec, *, suffix: str = "") -> str:
    """Derive an output filename from the input, never from raw user text."""
    stem = Path(display_name(loaded.source.value)).stem if loaded.source.kind == "path" else "image"
    if loaded.source.kind != "path":
        stem = "image"
    tag = f"-{suffix}" if suffix else ""
    return safe_filename(f"{stem}{tag}", spec.extension)


def size_change_for(input_bytes: int | None, output_bytes: int | None) -> SizeChange:
    return SizeChange(
        inputBytes=input_bytes,
        outputBytes=output_bytes,
        savedBytes=(input_bytes - output_bytes) if input_bytes and output_bytes else None,
        savedPercent=(round((1 - output_bytes / input_bytes) * 100, 2) if input_bytes and output_bytes else None),
    )


def target_size(image_size: tuple[int, int]) -> Size:
    return Size(image_size[0], image_size[1])


__all__ = [
    "RESOURCE_PREFIX",
    "RESOURCE_SCHEME",
    "ResultBuilder",
    "StoredOutput",
    "build_result",
    "default_filename",
    "human_bytes",
    "size_change_for",
    "target_size",
]

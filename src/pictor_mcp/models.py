"""Public result schema.

Everything a tool returns is described here. The shapes are deliberately
verbose but flat and self-describing, because four different consumers read
them:

* a **model** reads the text block, so the summary has to stand alone;
* a **structured client** reads ``structuredContent``, so the JSON has to be
  stable and complete;
* a **vision model** reads the inline ``image`` block, so the bytes have to be
  present when the caller asked for them;
* a **web UI** reads a ``resource_link`` or a signed URL, so the same artefact
  needs a fetchable address.

Field names are camelCase on the wire: JSON Schema keywords and MCP itself use
that convention, and mixing styles in one payload is a needless source of bugs
for client authors.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class FileOutput(_Model):
    """One produced artefact."""

    name: str = Field(description="Display filename.")
    path: str = Field(description="Path relative to the server's output root.")
    mime_type: str = Field(alias="mimeType")
    format: str = Field(description="Canonical codec name, e.g. 'webp'.")
    byte_size: int = Field(alias="byteSize")
    width: int
    height: int
    sha256: str = Field(description="SHA-256 of the encoded bytes.")
    url: str | None = Field(default=None, description="Signed URL, when output serving is enabled.")
    base64: str | None = Field(
        default=None,
        description="Base64 payload, present only when return_base64 was requested.",
    )


class SizeChange(_Model):
    """Before/after size accounting for one operation."""

    input_bytes: int | None = Field(default=None, alias="inputBytes")
    output_bytes: int | None = Field(default=None, alias="outputBytes")
    saved_bytes: int | None = Field(default=None, alias="savedBytes")
    saved_percent: float | None = Field(default=None, alias="savedPercent")


class ImageResult(_Model):
    """Standard envelope returned by every image tool.

    ``ok`` is always true on the success path; failures are reported through
    the MCP error channel instead, so a client never has to distinguish
    "failed" from "empty".
    """

    ok: bool = True
    operation: str = Field(description="Tool that produced this result.")
    outputs: list[FileOutput] = Field(default_factory=list)
    input: dict[str, Any] | None = Field(default=None, description="Summary of the source image, when one was read.")
    size_change: SizeChange | None = Field(default=None, alias="sizeChange")
    notes: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] | None = Field(
        default=None, description="Operation-specific measurements (e.g. comparison scores)."
    )
    inline_image_included: bool = Field(
        default=False,
        alias="inlineImageIncluded",
        description="True when an image block accompanies this result for vision-capable clients.",
    )


class InfoResult(_Model):
    """Result of inspecting an image without modifying it."""

    ok: bool = True
    operation: str = "image_info"
    image: dict[str, Any]
    exif: dict[str, Any] | None = None
    hashes: dict[str, str] | None = None
    dominant_colours: list[dict[str, Any]] | None = Field(default=None, alias="dominantColours")
    notes: list[str] = Field(default_factory=list)


class BatchFileResult(_Model):
    """Per-file outcome inside a batch run."""

    source: str
    status: Literal["ok", "failed", "skipped"]
    outputs: list[FileOutput] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    size_change: SizeChange | None = Field(default=None, alias="sizeChange")


class BatchResult(_Model):
    """Result of a batch operation."""

    ok: bool = True
    operation: str = "image_batch"
    processed: int
    succeeded: int
    failed: int
    total_input_bytes: int = Field(alias="totalInputBytes")
    total_output_bytes: int = Field(alias="totalOutputBytes")
    files: list[BatchFileResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    inline_image_included: bool = Field(default=False, alias="inlineImageIncluded")


class InputFile(_Model):
    """One file that is available to read, as returned by image_list_inputs."""

    path: str = Field(
        description=(
            "Pass this as `path` to any image tool. Relative to an input root where "
            "that is unambiguous, absolute otherwise."
        )
    )
    name: str = Field(description="Filename without its directory.")
    byte_size: int = Field(alias="byteSize")
    modified: str = Field(description="Last modification time, ISO-8601 UTC.")
    root: str = Field(description="The input root this file was found under.")


class InputListing(_Model):
    """What is available to read.

    Deliberately metadata only: listing must not decode anything, so it stays
    cheap enough to call before every operation, and cannot be used to make the
    server do work by remote control.
    """

    files: list[InputFile] = Field(default_factory=list)
    roots: list[str] = Field(default_factory=list, description="Roots that were searched, in order.")
    scanned: int = Field(description="Directory entries examined.")
    truncated: bool = Field(
        default=False,
        description="True when the scan or the result limit cut the listing short.",
    )


class BackendReport(_Model):
    """Runtime capability report."""

    ok: bool = True
    operation: str = "image_capabilities"
    server: dict[str, Any]
    protocol: dict[str, Any]
    formats: dict[str, Any]
    operations: list[dict[str, Any]]
    limits: dict[str, Any]
    security: dict[str, Any]
    backends: dict[str, Any]
    gpu: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


__all__ = [
    "BackendReport",
    "BatchFileResult",
    "BatchResult",
    "FileOutput",
    "ImageResult",
    "InfoResult",
    "InputFile",
    "InputListing",
    "SizeChange",
]

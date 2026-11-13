"""Public result schemas shared by tools and structured clients."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class FileOutput(_Model):
    """One produced artefact."""

    name: str = Field(description="Filename.")
    path: str = Field(description="Output-root-relative path.")
    mime_type: str = Field(alias="mimeType")
    format: str = Field(description="Output codec.")
    byte_size: int = Field(alias="byteSize")
    width: int
    height: int
    sha256: str = Field(description="SHA-256 of encoded bytes.")
    url: str | None = Field(default=None, description="Signed URL when output serving is enabled.")
    base64: str | None = Field(
        default=None,
        description="Encoded bytes, included when requested.",
    )


class SizeChange(_Model):
    """Before/after byte counts."""

    input_bytes: int | None = Field(default=None, alias="inputBytes")
    output_bytes: int | None = Field(default=None, alias="outputBytes")
    saved_bytes: int | None = Field(default=None, alias="savedBytes")
    saved_percent: float | None = Field(default=None, alias="savedPercent")


class ImageResult(_Model):
    """Shared result envelope for image tools."""

    ok: bool = True
    operation: str = Field(description="Producing tool.")
    outputs: list[FileOutput] = Field(default_factory=list)
    input: dict[str, Any] | None = Field(default=None, description="Source image summary.")
    size_change: SizeChange | None = Field(default=None, alias="sizeChange")
    notes: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] | None = Field(default=None, description="Operation metrics.")
    inline_image_included: bool = Field(
        default=False,
        alias="inlineImageIncluded",
        description="Whether the result includes an image block.",
    )


class InfoResult(_Model):
    """Image inspection result."""

    ok: bool = True
    operation: str = "image_info"
    image: dict[str, Any]
    exif: dict[str, Any] | None = None
    hashes: dict[str, str] | None = None
    dominant_colours: list[dict[str, Any]] | None = Field(default=None, alias="dominantColours")
    notes: list[str] = Field(default_factory=list)


class BatchFileResult(_Model):
    """One batch file outcome."""

    source: str
    status: Literal["ok", "failed", "skipped"]
    outputs: list[FileOutput] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    size_change: SizeChange | None = Field(default=None, alias="sizeChange")


class BatchResult(_Model):
    """Batch result."""

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
    """Readable input file."""

    path: str = Field(description=("Pass as `path` to an image tool."))
    name: str = Field(description="Filename.")
    byte_size: int = Field(alias="byteSize")
    modified: str = Field(description="Modified time (UTC).")
    root: str = Field(description="Input root.")


class InputListing(_Model):
    """Readable input listing."""

    files: list[InputFile] = Field(default_factory=list)
    roots: list[str] = Field(default_factory=list, description="Searched input roots.")
    scanned: int = Field(description="Entries scanned.")
    truncated: bool = Field(
        default=False,
        description="Whether the listing was truncated.",
    )


class BackendReport(_Model):
    """Server capabilities."""

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

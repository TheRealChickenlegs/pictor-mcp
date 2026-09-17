"""Server configuration, parsed once from the environment.

All settings use the ``PICTOR_`` prefix. Secure defaults are chosen so that a
bare ``docker run`` with no environment at all still refuses to read or write
anything outside ``/data``, never fetches URLs, and never binds a public
interface.

Parsing is strict: an unparseable or out-of-range value raises
:class:`~pictor_mcp.errors.ConfigError` at startup rather than silently
falling back, because a silently-ignored security setting is worse than a
server that refuses to boot.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

from .errors import ConfigError

_TRUE: Final = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSE: Final = frozenset({"0", "false", "no", "off", "n", "f", ""})

#: Hosts that are loopback-only. When the bind address is one of these we
#: derive a tight allowed-host/Origin list instead of asking the operator.
_LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "::1", "localhost"})


class _Env:
    """Typed, fail-loud accessor over a string mapping."""

    __slots__ = ("_source",)

    def __init__(self, source: Mapping[str, str]) -> None:
        self._source = source

    def raw(self, name: str, default: str | None = None) -> str | None:
        value = self._source.get(name)
        if value is None:
            return default
        value = value.strip()
        return value if value else default

    def bool(self, name: str, default: bool) -> bool:
        raw = self.raw(name)
        if raw is None:
            return default
        lowered = raw.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ConfigError(f"{name} must be a boolean (got {raw!r})")

    def int(self, name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
        raw = self.raw(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer (got {raw!r})") from exc
        if minimum is not None and value < minimum:
            raise ConfigError(f"{name} must be >= {minimum} (got {value})")
        if maximum is not None and value > maximum:
            raise ConfigError(f"{name} must be <= {maximum} (got {value})")
        return value

    def float(self, name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
        raw = self.raw(name)
        if raw is None:
            return default
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be a number (got {raw!r})") from exc
        if minimum is not None and value < minimum:
            raise ConfigError(f"{name} must be >= {minimum} (got {value})")
        if maximum is not None and value > maximum:
            raise ConfigError(f"{name} must be <= {maximum} (got {value})")
        return value

    def list(self, name: str) -> list[str]:
        """Comma- or whitespace-separated list, empties dropped."""
        raw = self.raw(name)
        if raw is None:
            return []
        return [part for part in re.split(r"[,\s]+", raw) if part]

    def paths(self, name: str) -> list[Path]:
        resolved: list[Path] = []
        for entry in self.list(name):
            path = Path(entry)
            if not path.is_absolute():
                raise ConfigError(f"{name} entries must be absolute paths (got {entry!r})")
            try:
                resolved.append(path.resolve())
            except OSError as exc:  # pragma: no cover - exotic filesystem failure
                raise ConfigError(f"{name} entry {entry!r} could not be resolved: {exc}") from exc
        return resolved


@dataclass(frozen=True, slots=True)
class Limits:
    """Resource ceilings applied before and during every decode."""

    #: Largest accepted input file, in bytes.
    max_file_bytes: int = 64 * 1024 * 1024
    #: Largest accepted decoded pixel count. Guards decompression bombs.
    max_pixels: int = 64_000_000
    #: Largest accepted single dimension (width or height).
    max_dimension: int = 24_000
    #: Largest accepted animation frame count.
    max_frames: int = 120
    #: Largest accepted *total* pixel count across every animation frame.
    #:
    #: `max_pixels` bounds one frame, and `max_frames` bounds how many there
    #: are, but neither bounds the product: a 2000x2000 120-frame GIF passes
    #: both while decoding to ~1.9 GB, because every frame is held as a full
    #: RGBA bitmap before encoding. This is the limit that bounds that product.
    max_animation_pixels: int = 128_000_000
    #: Largest accepted output pixel count after an operation.
    max_output_pixels: int = 256_000_000
    #: Cap on how much EXIF we will carry across a transform.
    max_exif_bytes: int = 64 * 1024
    #: Wall-clock budget for a single tool call, checked between pipeline steps
    #: and between quality-search probes. Cooperative: a single Pillow call is
    #: not interruptible from Python, but a long sequence of them is stopped.
    op_timeout_seconds: float = 120.0
    #: Concurrent image operations.
    max_concurrency: int = 4
    #: Files touched by one ``image_batch`` call.
    max_batch_files: int = 64
    #: Largest inline base64 image embedded in a tool *result*.
    max_inline_image_bytes: int = 1024 * 1024
    #: Largest base64 image accepted as tool *input*.
    max_input_base64_bytes: int = 48 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    """Outbound HTTP policy for fetching images by URL."""

    enabled: bool = False
    #: Empty means "any public host"; private/loopback/link-local stay blocked.
    allowed_hosts: tuple[str, ...] = ()
    allowed_ports: tuple[int, ...] = (80, 443)
    max_bytes: int = 32 * 1024 * 1024
    max_redirects: int = 3
    timeout_seconds: float = 15.0
    #: Certificate verification for HTTPS. Only ever disabled for a pinned
    #: internal mirror; documented as a downgrade.
    verify_tls: bool = True


@dataclass(frozen=True, slots=True)
class HttpPolicy:
    """HTTP transport hardening."""

    host: str = "127.0.0.1"
    port: int = 8077
    path: str = "/mcp"
    stateless_http: bool = True
    json_response: bool = False
    #: Reject requests whose Host/Origin do not match the allow-list. This is
    #: the DNS-rebinding defence the MCP spec requires of local servers.
    dns_rebinding_protection: bool = True
    allowed_hosts: tuple[str, ...] = ()
    allowed_origins: tuple[str, ...] = ()
    #: Maximum accepted request body. Image servers legitimately need more than
    #: the SDK's 4 MiB default; the cap still bounds memory per request.
    max_request_body_bytes: int = 48 * 1024 * 1024
    #: Optional pre-shared bearer token. Empty disables auth (LAN-only mode).
    auth_token: str = ""
    #: Serve generated files over HTTP so web UIs can render them directly.
    serve_outputs: bool = False
    #: Public base URL used to build absolute links (behind a reverse proxy).
    public_base_url: str = ""
    #: HMAC secret for signed, expiring output URLs.
    url_secret: str = ""
    url_ttl_seconds: int = 3600
    #: Emit HSTS. Only meaningful when TLS terminates in front of us.
    enable_hsts: bool = False
    #: Emit uvicorn's per-request access log.
    #:
    #: Off by default because the signed output URLs carry their HMAC signature
    #: in the query string, so an access log line is a usable credential for as
    #: long as the link lives. Application logging (tool calls, rejections) is
    #: unaffected and stays on.
    access_log: bool = False


@dataclass(frozen=True, slots=True)
class Config:
    """Fully resolved configuration."""

    transport: str
    limits: Limits
    fetch: FetchPolicy
    http: HttpPolicy
    #: Directories that may be read from.
    input_roots: tuple[Path, ...]
    #: The single directory that may be written to.
    output_root: Path
    #: Strip EXIF/ICC/other metadata from outputs unless asked to keep it.
    strip_metadata: bool = True
    #: GPU policy: ``auto`` uses a CUDA backend only when one is importable.
    gpu: str = "auto"
    #: Only use the GPU above this pixel count (transfer overhead dominates
    #: for small images).
    gpu_min_pixels: int = 4_000_000
    #: Embed the resulting image inline in tool results by default.
    inline_images: bool = True
    log_level: str = "INFO"
    #: Background-removal model name passed to rembg.
    bg_model: str = "u2net"
    #: Directories scanned for fonts available to the text watermark tool.
    #: Fonts are selected by *name* from this index, never by caller-supplied
    #: path, so watermarking does not become another arbitrary-file-read.
    font_dirs: tuple[Path, ...] = ()
    extras: dict[str, str] = field(default_factory=dict)

    def describe_roots(self) -> dict[str, list[str]]:
        return {
            "input_roots": [str(p) for p in self.input_roots],
            "output_root": str(self.output_root),
        }


def _derive_allowed_hosts(host: str, configured: list[str], port: int) -> tuple[str, ...]:
    """Build a Host allow-list.

    When the operator gives an explicit list we honour it verbatim. Otherwise we
    derive the narrowest list that still works: loopback binds get loopback
    names only, and a wildcard bind gets a permissive pattern (still rejecting a
    *missing* Host header). Operators exposing this to a browser should set
    ``PICTOR_ALLOWED_HOSTS`` explicitly.
    """
    if configured:
        return tuple(configured)
    if host in _LOOPBACK_HOSTS:
        return (
            "127.0.0.1:*",
            "localhost:*",
            "[::1]:*",
            f"127.0.0.1:{port}",
            f"localhost:{port}",
        )
    return ("*",)


def _derive_allowed_origins(host: str, configured: list[str]) -> tuple[str, ...]:
    if configured:
        return tuple(configured)
    if host in _LOOPBACK_HOSTS:
        return ("http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*")
    # An empty Origin allow-list means "no browser Origin is acceptable", which
    # is exactly right for a wildcard bind: MCP clients send no Origin header
    # (always permitted), while a cross-site browser request is rejected. This
    # closes the DNS-rebinding hole without breaking CLI clients.
    return ()


#: A Host header is `host[:port]` and never carries a scheme; an Origin always
#: carries one. An entry in the wrong list therefore matches nothing at all -
#: which is silent, and looks exactly like protection that is merely not needed
#: yet. `PICTOR_FETCH_ALLOWED_HOSTS` is compared against the hostname alone, so
#: a port there is inert too (the port is checked against
#: PICTOR_FETCH_ALLOWED_PORTS instead). These are reported at startup and by
#: `--check`, because a pattern that cannot match is worth knowing about before
#: a client is refused.
def inert_allow_list_entries(config: Config) -> list[str]:
    """Allow-list entries that can never match their header, with the reason."""
    messages: list[str] = []

    for entry in config.http.allowed_hosts:
        if "://" in entry:
            messages.append(
                f"PICTOR_ALLOWED_HOSTS entry {entry!r} contains a scheme and can never match a Host "
                "header, which is only host[:port]; that form belongs in PICTOR_ALLOWED_ORIGINS"
            )
    for entry in config.http.allowed_origins:
        if entry == "*" or entry.startswith("*.") or "://" in entry:
            continue
        messages.append(
            f"PICTOR_ALLOWED_ORIGINS entry {entry!r} has no scheme and can never match an Origin, "
            "which always looks like http://host[:port]; that form belongs in PICTOR_ALLOWED_HOSTS"
        )
    for entry in config.fetch.allowed_hosts:
        if "://" in entry or _names_a_port(entry):
            messages.append(
                f"PICTOR_FETCH_ALLOWED_HOSTS entry {entry!r} is matched against the hostname alone, so "
                "a scheme or a port there can never match; use the bare name and "
                "PICTOR_FETCH_ALLOWED_PORTS for the port"
            )
    if config.http.public_base_url and not config.http.serve_outputs:
        messages.append(
            "PICTOR_PUBLIC_BASE_URL is set but PICTOR_SERVE_OUTPUTS=false, so results carry no link and "
            "a chat UI has nothing to render; set PICTOR_SERVE_OUTPUTS=true or drop the base URL"
        )
    if config.http.serve_outputs and config.http.dns_rebinding_protection and config.http.public_base_url:
        public_host = urlsplit(config.http.public_base_url).hostname
        if public_host and not _host_list_permits(public_host, config.http.allowed_hosts):
            messages.append(
                f"PICTOR_PUBLIC_BASE_URL names {public_host!r}, which PICTOR_ALLOWED_HOSTS does not admit; "
                "if your reverse proxy forwards that Host header unchanged - the usual arrangement - the "
                "browser's request for a generated file link is refused by this server"
            )
    return messages


def _host_list_permits(host: str, patterns: tuple[str, ...]) -> bool:
    """Whether an HTTP Host allow-list would admit ``host``.

    Uses the guard's own matcher rather than a second copy of its semantics, so
    a pattern this server accepts cannot be reported as broken. It is
    deliberately forgiving in one direction: a pattern naming a port
    (``pictor.example.com:8077``) still admits the port-less Host header a proxy
    sends for the public URL. This only ever feeds a warning, and a false alarm
    would train the reader to ignore the real ones.
    """
    if not patterns:
        return False

    from .security.auth import _host_matches

    if _host_matches(host, patterns):
        return True
    for pattern in patterns:
        base, separator, _port = pattern.lower().rpartition(":")
        if separator and base.strip("[]") == host:
            return True
    return False


def _names_a_port(entry: str) -> bool:
    """True for ``host:port`` and ``[v6]:port``, false for a bare name or ``[v6]``."""
    if entry.startswith("["):
        return "]:" in entry
    return ":" in entry


def _parse_ports(raw: str | None) -> list[int]:
    ports: list[int] = []
    for token in (raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            port = int(token)
        except ValueError as exc:
            raise ConfigError(f"PICTOR_FETCH_ALLOWED_PORTS invalid port {token!r}") from exc
        if not 1 <= port <= 65535:
            raise ConfigError(f"PICTOR_FETCH_ALLOWED_PORTS port out of range: {port}")
        ports.append(port)
    return ports or [80, 443]


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Parse configuration from ``env`` (defaults to ``os.environ``)."""
    e = _Env(os.environ if env is None else env)

    transport = (e.raw("PICTOR_TRANSPORT", "stdio") or "stdio").lower()
    if transport not in {"stdio", "streamable-http", "sse"}:
        raise ConfigError(f"PICTOR_TRANSPORT must be one of stdio, streamable-http, sse (got {transport!r})")

    limits = Limits(
        max_file_bytes=e.int("PICTOR_MAX_FILE_BYTES", 64 * 1024 * 1024, minimum=1024),
        max_pixels=e.int("PICTOR_MAX_PIXELS", 64_000_000, minimum=10_000),
        max_dimension=e.int("PICTOR_MAX_DIMENSION", 24_000, minimum=16),
        max_frames=e.int("PICTOR_MAX_FRAMES", 120, minimum=1),
        max_animation_pixels=e.int("PICTOR_MAX_ANIMATION_PIXELS", 128_000_000, minimum=10_000),
        max_output_pixels=e.int("PICTOR_MAX_OUTPUT_PIXELS", 256_000_000, minimum=10_000),
        max_exif_bytes=e.int("PICTOR_MAX_EXIF_BYTES", 64 * 1024, minimum=0),
        op_timeout_seconds=e.float("PICTOR_OP_TIMEOUT_SECONDS", 120.0, minimum=1.0),
        max_concurrency=e.int("PICTOR_MAX_CONCURRENCY", 4, minimum=1, maximum=64),
        max_batch_files=e.int("PICTOR_MAX_BATCH_FILES", 64, minimum=1, maximum=10_000),
        max_inline_image_bytes=e.int("PICTOR_MAX_INLINE_IMAGE_BYTES", 1024 * 1024, minimum=0),
        max_input_base64_bytes=e.int("PICTOR_MAX_INPUT_BASE64_BYTES", 48 * 1024 * 1024, minimum=1024),
    )

    input_roots = e.paths("PICTOR_INPUT_ROOTS") or [Path("/data")]
    output_root = (e.paths("PICTOR_OUTPUT_ROOT") or [Path("/data/output")])[0]

    if output_root in input_roots:
        raise ConfigError(
            "PICTOR_OUTPUT_ROOT must not also be an input root; keep writable output separate from read-only inputs"
        )
    if any(output_root == root for root in input_roots):
        raise ConfigError("PICTOR_OUTPUT_ROOT overlaps an input root")

    fetch = FetchPolicy(
        enabled=e.bool("PICTOR_ALLOW_NET_FETCH", False),
        allowed_hosts=tuple(h.lower() for h in e.list("PICTOR_FETCH_ALLOWED_HOSTS")),
        allowed_ports=tuple(_parse_ports(e.raw("PICTOR_FETCH_ALLOWED_PORTS", "80,443"))),
        max_bytes=e.int("PICTOR_FETCH_MAX_BYTES", 32 * 1024 * 1024, minimum=1024),
        max_redirects=e.int("PICTOR_FETCH_MAX_REDIRECTS", 3, minimum=0, maximum=10),
        timeout_seconds=e.float("PICTOR_FETCH_TIMEOUT_SECONDS", 15.0, minimum=1.0, maximum=300.0),
        verify_tls=e.bool("PICTOR_FETCH_VERIFY_TLS", True),
    )

    host = e.raw("PICTOR_HOST", "127.0.0.1") or "127.0.0.1"
    port = e.int("PICTOR_PORT", 8077, minimum=1, maximum=65535)
    path = e.raw("PICTOR_STREAMABLE_HTTP_PATH", "/mcp") or "/mcp"
    if not path.startswith("/"):
        raise ConfigError(f"PICTOR_STREAMABLE_HTTP_PATH must start with '/' (got {path!r})")

    auth_token = e.raw("PICTOR_AUTH_TOKEN", "") or ""
    if auth_token and len(auth_token) < 16:
        raise ConfigError("PICTOR_AUTH_TOKEN must be at least 16 characters; generate one with `openssl rand -hex 32`")

    serve_outputs = e.bool("PICTOR_SERVE_OUTPUTS", False)
    public_base_url = (e.raw("PICTOR_PUBLIC_BASE_URL", "") or "").rstrip("/")
    if serve_outputs and public_base_url and not public_base_url.startswith(("http://", "https://")):
        raise ConfigError("PICTOR_PUBLIC_BASE_URL must start with http:// or https://")

    url_secret = e.raw("PICTOR_URL_SECRET", "") or auth_token
    # Serving generated files with neither a bearer token nor a signing secret
    # would make every image the server has ever produced readable by anyone who
    # can reach the port. Refuse to start rather than fail open.
    if serve_outputs and not auth_token and not url_secret:
        raise ConfigError(
            "PICTOR_SERVE_OUTPUTS=true requires PICTOR_AUTH_TOKEN or PICTOR_URL_SECRET; "
            "otherwise generated files would be downloadable without authentication"
        )
    if url_secret and len(url_secret) < 16:
        raise ConfigError("PICTOR_URL_SECRET must be at least 16 characters")

    http = HttpPolicy(
        host=host,
        port=port,
        path=path,
        stateless_http=e.bool("PICTOR_STATELESS_HTTP", True),
        json_response=e.bool("PICTOR_JSON_RESPONSE", False),
        dns_rebinding_protection=e.bool("PICTOR_DNS_REBINDING_PROTECTION", True),
        allowed_hosts=_derive_allowed_hosts(host, e.list("PICTOR_ALLOWED_HOSTS"), port),
        allowed_origins=_derive_allowed_origins(host, e.list("PICTOR_ALLOWED_ORIGINS")),
        max_request_body_bytes=e.int("PICTOR_MAX_REQUEST_BODY_BYTES", 48 * 1024 * 1024, minimum=1024),
        auth_token=auth_token,
        serve_outputs=serve_outputs,
        public_base_url=public_base_url,
        url_secret=url_secret,
        url_ttl_seconds=e.int("PICTOR_URL_TTL_SECONDS", 3600, minimum=60, maximum=30 * 24 * 3600),
        enable_hsts=e.bool("PICTOR_ENABLE_HSTS", False),
        access_log=e.bool("PICTOR_HTTP_ACCESS_LOG", False),
    )

    gpu = (e.raw("PICTOR_GPU", "auto") or "auto").lower()
    if gpu not in {"auto", "off", "torch", "cuda"}:
        raise ConfigError(f"PICTOR_GPU must be one of auto, off, torch (got {gpu!r})")

    log_level = (e.raw("PICTOR_LOG_LEVEL", "INFO") or "INFO").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError(f"PICTOR_LOG_LEVEL invalid (got {log_level!r})")

    return Config(
        transport=transport,
        limits=limits,
        fetch=fetch,
        http=http,
        input_roots=tuple(input_roots),
        output_root=output_root,
        strip_metadata=e.bool("PICTOR_STRIP_METADATA", True),
        gpu=gpu,
        gpu_min_pixels=e.int("PICTOR_GPU_MIN_PIXELS", 4_000_000, minimum=0),
        inline_images=e.bool("PICTOR_INLINE_IMAGES", True),
        log_level=log_level,
        bg_model=e.raw("PICTOR_BG_MODEL", "u2net") or "u2net",
        font_dirs=tuple(e.paths("PICTOR_FONT_DIRS") or [Path("/usr/share/fonts")]),
    )


def is_loopback_bind(host: str) -> bool:
    return host in _LOOPBACK_HOSTS


__all__ = [
    "Config",
    "FetchPolicy",
    "HttpPolicy",
    "Limits",
    "is_loopback_bind",
    "load_config",
]

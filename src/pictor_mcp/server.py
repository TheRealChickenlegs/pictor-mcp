"""Server entry point: configuration, wiring, transports.

Three transports are supported and all three are served by the same server
object and the same tool set:

``stdio``
    For clients that launch the process themselves (OpenCode, Hermes in local
    mode, Claude Desktop, DSH configured with a command).
``streamable-http``
    For Docker and every network client (Open WebUI, DSH remote, Hermes remote).
    The MCP Python SDK negotiates per connection: a modern 2026-07-28 request is
    served statelessly with no handshake, while a legacy client that opens with
    ``initialize`` gets the classic session behaviour. One endpoint therefore
    works for both generations of client without a compatibility flag.
``sse``
    The deprecated HTTP+SSE transport, kept only for older clients.

Everything security-relevant is assembled here and passed down, so a tool
cannot accidentally run without confinement: there is exactly one
:class:`~pictor_mcp.security.paths.PathJail` and one
:class:`~pictor_mcp.security.ratelimit.ConcurrencyGate`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from . import __version__
from .backends import build_registry
from .config import Config, load_config
from .errors import ConfigError, PictorError
from .imaging.fonts import FontIndex
from .imaging.loader import ImageLoader
from .outputs import ResultBuilder
from .resources import register_resources, serve_mime
from .security.auth import (
    BearerAuthMiddleware,
    HostOriginGuardMiddleware,
    SecurityHeadersMiddleware,
    verify_signed_path,
)
from .security.limits import configure_pillow
from .security.net import SafeFetcher
from .security.paths import PathJail
from .security.ratelimit import ConcurrencyGate
from .tools.context import ToolContext

logger = logging.getLogger("pictor_mcp")

#: Endpoints served without authentication. ``/healthz`` is deliberately
#: information-free so a container healthcheck needs no credentials.
_AUTH_EXEMPT = ("/healthz",)

INSTRUCTIONS = """\
Image operations server. It can inspect, convert, resize, compress, crop, rotate,
watermark, optimise and background-remove images, and run the same pipeline over
many files at once.

How to use it well:
- Prefer `image_transform` when you need more than one step: pass an ordered
  `operations` list and the image crosses the wire only once.
- Use `image_capabilities` if you are unsure which formats or features exist.
- Input can be a `path` inside the server's input root, `base64_data`, or a `url`
  (URL fetching may be disabled). Supply exactly one.
- Every result is written to the server's output root and described in the
  response. The text block always states the relative path, dimensions, byte
  sizes and any saving, so you do not need to re-inspect the file.
- Results also include `structuredContent` for programmatic use, and an inline
  image when `return_image` is true so vision-capable clients can see the output.
- Metadata (EXIF, GPS, ICC) is stripped by default; pass `strip_metadata=false`
  only when a downstream step genuinely needs it.
- Use `image_compare` to verify that a transform did what you intended.
"""


# --------------------------------------------------------------------- logging


class _RedactingFormatter(logging.Formatter):
    """Keeps secrets and image payloads out of the logs."""

    _REPLACEMENTS = (
        ("PICTOR_AUTH_TOKEN", "***"),
        ("PICTOR_URL_SECRET", "***"),
    )

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for key, mask in self._REPLACEMENTS:
            value = os.environ.get(key)
            if value and len(value) >= 8:
                text = text.replace(value, mask)
        return text


def configure_logging(level: str) -> None:
    """Log to stderr only.

    On stdio transports stdout is the protocol channel: a stray print or log
    line there corrupts the JSON-RPC stream, so nothing is ever written to it.
    """
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))


# ------------------------------------------------------------------ assembling


def _verify_output_writable(root: Path) -> None:
    """Prove at startup that the output root can actually be written to.

    ``mkdir(exist_ok=True)`` succeeds on an existing but unwritable directory, so
    a container started against a root-owned volume would boot cleanly and then
    fail every single tool call. Failing here turns a confusing runtime error
    into an immediate, actionable one - the usual cause is a host directory
    owned by the wrong uid.
    """
    probe = root / f".pictor-write-probe-{os.getpid()}"
    fd = -1
    try:
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise ConfigError(
            f"output root {root} is not writable by uid {os.getuid()}: {exc.strerror}. "
            "In Docker, make sure the mounted host directory is owned by the container "
            "user (uid 10001)."
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(probe)


def build_context(config: Config) -> ToolContext:
    """Construct the single, shared, fully-confined tool context."""
    configure_pillow(config.limits)

    jail = PathJail(config.input_roots, config.output_root)
    try:
        jail.ensure_output_root()
    except OSError as exc:
        raise ConfigError(f"output root could not be created: {exc}") from exc
    _verify_output_writable(config.output_root)

    loader = ImageLoader(config, jail, SafeFetcher(config.fetch))
    builder = ResultBuilder(config, jail)
    fonts = FontIndex(config.font_dirs)
    registry = build_registry(config)
    gate = ConcurrencyGate(
        config.limits.max_concurrency,
        timeout_seconds=min(60.0, config.limits.op_timeout_seconds),
    )
    return ToolContext(
        config=config,
        jail=jail,
        loader=loader,
        builder=builder,
        fonts=fonts,
        registry=registry,
        gate=gate,
        server_version=__version__,
    )


def build_server(config: Config, ctx: ToolContext):
    """Create the MCP server and register every tool and resource."""
    from mcp.server.mcpserver import MCPServer

    from .tools import register_all

    server = MCPServer(
        name="pictor-mcp",
        title="Pictor image operations",
        description=(
            "Secure image processing: convert, resize, compress, crop, rotate, watermark, "
            "background-remove, batch and optimise for the web."
        ),
        instructions=INSTRUCTIONS,
        version=__version__,
        log_level=config.log_level,
    )
    register_all(server, ctx)
    register_resources(server, ctx)
    return server


# ------------------------------------------------------------------ HTTP extras


def _files_endpoint(ctx: ToolContext):
    """Serve generated files so web UIs can render them directly.

    Disabled unless ``PICTOR_SERVE_OUTPUTS=true``, and configuration forbids
    enabling it without either a bearer token or a signing secret, so this can
    never become an open directory of everything the server has produced.
    """

    async def serve(request: Request) -> Response:
        relative = request.path_params.get("path", "")
        secret = ctx.config.http.url_secret

        if secret:
            expires = request.query_params.get("e", "")
            signature = request.query_params.get("s", "")
            if not verify_signed_path(secret, relative, expires, signature):
                return JSONResponse(
                    {"error": "invalid or expired link"},
                    status_code=403,
                    headers={"cache-control": "no-store"},
                )

        try:
            import anyio

            data = await anyio.to_thread.run_sync(ctx.jail.read_output_bytes, relative)
        except PictorError as exc:
            return JSONResponse({"error": exc.message, "code": exc.code}, status_code=exc.http_status)

        content_type, inline = serve_mime(relative)

        return Response(
            content=data,
            media_type=content_type,
            headers={
                "cache-control": f"private, max-age={min(ctx.config.http.url_ttl_seconds, 86400)}",
                "content-disposition": "inline" if inline else "attachment",
                "x-content-type-options": "nosniff",
                # Belt and braces: even if a response were somehow typed as
                # HTML, this policy gives a document no way to load anything or
                # run script from the server's origin.
                "content-security-policy": "default-src 'none'; sandbox",
            },
        )

    return serve


async def _healthz(_request: Request) -> Response:
    return JSONResponse({"status": "ok"})


def _attach_http_extras(app, config: Config, ctx: ToolContext) -> None:
    """Add health, file serving, and the security middleware stack."""
    app.routes.append(Route("/healthz", _healthz, methods=["GET"]))

    if config.http.serve_outputs:
        app.routes.append(Route("/files/{path:path}", _files_endpoint(ctx), methods=["GET"]))
        logger.info("output file serving enabled at /files/")
    else:
        logger.info("output file serving disabled (set PICTOR_SERVE_OUTPUTS=true to enable)")

    if config.http.auth_token:
        app.add_middleware(
            BearerAuthMiddleware,
            token=config.http.auth_token,
            exempt_paths=_AUTH_EXEMPT,
            # /files/ URLs carry their own credential: an HMAC signature with an
            # expiry scoped to one object. That is what lets a browser <img> tag
            # load a generated image without ever holding the API token.
            # Signing is always active there, because configuration refuses to
            # enable output serving without a token or a signing secret.
            exempt_prefixes=("/files/",) if config.http.serve_outputs else (),
        )
        logger.info("bearer token authentication enabled")
    else:
        logger.warning(
            "no PICTOR_AUTH_TOKEN set: the HTTP endpoint is unauthenticated. "
            "Only expose it on a trusted network or behind an authenticating proxy."
        )

    # Middleware added later wraps earlier ones, so the effective request order is
    # SecurityHeaders -> HostOriginGuard -> BearerAuth -> routes. Host/Origin is
    # checked before credentials are examined, and the headers are added last so
    # even a 401 or 403 carries them.
    app.add_middleware(
        HostOriginGuardMiddleware,
        allowed_hosts=config.http.allowed_hosts,
        allowed_origins=config.http.allowed_origins,
        enabled=config.http.dns_rebinding_protection,
    )
    if config.http.dns_rebinding_protection:
        logger.info(
            "DNS-rebinding protection enabled: allowed hosts=%s allowed origins=%s",
            ", ".join(config.http.allowed_hosts) or "<none>",
            ", ".join(config.http.allowed_origins) or "<none - browser origins refused>",
        )
    else:
        logger.warning(
            "PICTOR_DNS_REBINDING_PROTECTION=false: Host and Origin headers are not validated. "
            "Do not run this way on a host reachable from a browser."
        )

    app.add_middleware(SecurityHeadersMiddleware, hsts=config.http.enable_hsts)


def build_http_app(config: Config, ctx: ToolContext, server) -> Any:
    """Build the Streamable HTTP ASGI application."""
    from mcp.server.transport_security import TransportSecuritySettings

    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=config.http.dns_rebinding_protection,
        allowed_hosts=list(config.http.allowed_hosts),
        allowed_origins=list(config.http.allowed_origins),
    )
    app = server.streamable_http_app(
        streamable_http_path=config.http.path,
        stateless_http=config.http.stateless_http,
        json_response=config.http.json_response,
        max_request_body_size=config.http.max_request_body_bytes,
        transport_security=security,
        host=config.http.host,
    )
    _attach_http_extras(app, config, ctx)
    return app


def build_sse_app(config: Config, ctx: ToolContext, server) -> Any:
    """Build the deprecated HTTP+SSE application for older clients."""
    from mcp.server.transport_security import TransportSecuritySettings

    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=config.http.dns_rebinding_protection,
        allowed_hosts=list(config.http.allowed_hosts),
        allowed_origins=list(config.http.allowed_origins),
    )
    app = server.sse_app(
        max_request_body_size=config.http.max_request_body_bytes,
        transport_security=security,
        host=config.http.host,
    )
    _attach_http_extras(app, config, ctx)
    return app


# ----------------------------------------------------------------------- main


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pictor-mcp",
        description="Secure MCP server for image operations.",
    )
    parser.add_argument("--version", action="version", version=f"pictor-mcp {__version__}")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        help="Override PICTOR_TRANSPORT.",
    )
    parser.add_argument("--host", help="Override PICTOR_HOST.")
    parser.add_argument("--port", type=int, help="Override PICTOR_PORT.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration, print the resolved setup (secrets redacted) and exit.",
    )
    return parser.parse_args(argv)


def _resolved_config(args: argparse.Namespace) -> Config:
    env = dict(os.environ)
    if args.transport:
        env["PICTOR_TRANSPORT"] = args.transport
    if args.host:
        env["PICTOR_HOST"] = args.host
    if args.port:
        env["PICTOR_PORT"] = str(args.port)
    return load_config(env)


def _redacted_summary(config: Config, ctx: ToolContext) -> dict[str, Any]:
    return {
        "version": __version__,
        "transport": config.transport,
        "http": {
            "host": config.http.host,
            "port": config.http.port,
            "path": config.http.path,
            "statelessHttp": config.http.stateless_http,
            "authToken": "set" if config.http.auth_token else "NOT SET",
            "dnsRebindingProtection": config.http.dns_rebinding_protection,
            "allowedHosts": list(config.http.allowed_hosts),
            "allowedOrigins": list(config.http.allowed_origins),
            "maxRequestBodyBytes": config.http.max_request_body_bytes,
            "serveOutputs": config.http.serve_outputs,
            "publicBaseUrl": config.http.public_base_url or None,
        },
        "filesystem": {
            "inputRoots": [str(p) for p in config.input_roots],
            "outputRoot": str(config.output_root),
        },
        "limits": {
            "maxFileBytes": config.limits.max_file_bytes,
            "maxPixels": config.limits.max_pixels,
            "maxDimension": config.limits.max_dimension,
            "maxConcurrency": config.limits.max_concurrency,
            "maxBatchFiles": config.limits.max_batch_files,
        },
        "networkFetch": {
            "enabled": config.fetch.enabled,
            "allowedHosts": list(config.fetch.allowed_hosts) or "any public host",
        },
        "gpu": config.gpu,
        "acceleration": ctx.registry.active,
        "stripMetadataByDefault": config.strip_metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        config = _resolved_config(args)
    except ConfigError as exc:
        print(f"configuration error: {exc.message}", file=sys.stderr)
        return 2

    configure_logging(config.log_level)

    try:
        ctx = build_context(config)
    except ConfigError as exc:
        print(f"configuration error: {exc.message}", file=sys.stderr)
        return 2

    if args.check:
        print(json.dumps(_redacted_summary(config, ctx), indent=2))
        return 0

    server = build_server(config, ctx)
    logger.info(
        "pictor-mcp %s starting: transport=%s acceleration=%s",
        __version__,
        config.transport,
        ctx.registry.active,
    )

    if config.transport == "stdio":
        server.run("stdio")
        return 0

    import uvicorn

    if config.transport == "sse":
        app = build_sse_app(config, ctx, server)
        logger.warning("the HTTP+SSE transport is deprecated; prefer streamable-http")
    else:
        app = build_http_app(config, ctx, server)

    uvicorn.run(
        app,
        host=config.http.host,
        port=config.http.port,
        log_level=config.log_level.lower(),
        # Signed output URLs carry their HMAC signature in the query string, so
        # a per-request access log line is a credential while the link lives.
        # Application logs are unaffected.
        access_log=config.http.access_log,
        # Trust no proxy headers by default: honouring X-Forwarded-For from an
        # untrusted client would let it spoof its origin in our logs.
        proxy_headers=False,
        server_header=False,
        date_header=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

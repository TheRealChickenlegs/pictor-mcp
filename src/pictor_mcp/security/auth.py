"""HTTP transport security: bearer authentication and response hardening.

The MCP SDK ships OAuth 2.0 resource-server support, which is the right tool
for a public deployment but a poor fit for a LAN-only service behind a
pre-shared token: it requires an issuer URL, token introspection, and a client
registration story that a homelab does not have. What is offered here is the
simple, well-understood alternative, implemented carefully:

* constant-time token comparison, so the endpoint cannot be used as a timing
  oracle to recover the token byte by byte;
* tokens accepted from either ``Authorization: Bearer`` (the MCP norm) or
  ``X-API-Key`` (what several web UIs send);
* auth applied to *every* route including the file server and health check,
  with an explicit exemption list rather than an implicit one;
* signed, expiring URLs for generated files, so a link handed to a browser
  stops working and cannot be edited into a directory listing.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from collections.abc import Iterable
from typing import Any
from urllib.parse import quote

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

_UNAUTHORIZED_BODY = json.dumps(
    {
        "jsonrpc": "2.0",
        "error": {"code": -32001, "message": "Unauthorized: a valid bearer token is required"},
        "id": None,
    }
).encode()


def _extract_token(headers: Headers) -> str | None:
    authorization = headers.get("authorization")
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    api_key = headers.get("x-api-key")
    if api_key and api_key.strip():
        return api_key.strip()
    return None


class BearerAuthMiddleware:
    """Pure-ASGI bearer-token gate.

    Pure ASGI rather than ``BaseHTTPMiddleware`` so that it also covers
    WebSocket and lifespan scopes correctly and never buffers a streaming
    response.

    ``exempt_paths`` and ``exempt_prefixes`` exist for two different reasons.
    The health endpoint carries no information and must answer a container
    healthcheck without credentials. The ``/files/`` prefix is exempt because
    those URLs carry their own credential: an HMAC signature with an expiry,
    scoped to a single object. That is what lets a browser ``<img>`` tag load a
    generated image without ever seeing the API token; requiring the token there
    as well would make signed URLs pointless.
    """

    def __init__(
        self,
        app: ASGIApp,
        token: str,
        *,
        exempt_paths: Iterable[str] = (),
        exempt_prefixes: Iterable[str] = (),
    ) -> None:
        self.app = app
        self._token = token.encode()
        self._exempt = frozenset(exempt_paths)
        self._exempt_prefixes = tuple(exempt_prefixes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self._exempt or (self._exempt_prefixes and path.startswith(self._exempt_prefixes)):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        presented = _extract_token(headers)

        # Always run the comparison, even when nothing was presented, so the
        # work done does not reveal whether a token was supplied at all.
        candidate = (presented or "").encode()
        if not secrets.compare_digest(candidate, self._token) or presented is None:
            await self._reject(scope, receive, send)
            return

        await self.app(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        body = _UNAUTHORIZED_BODY
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b'Bearer realm="pictor-mcp"'),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class SecurityHeadersMiddleware:
    """Adds conservative security headers to every response."""

    def __init__(self, app: ASGIApp, *, hsts: bool = False) -> None:
        self.app = app
        self._hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                raw = list(message.get("headers", []))
                existing = {name.lower() for name, _ in raw}
                extra: list[tuple[bytes, bytes]] = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-frame-options", b"DENY"),
                    (b"cross-origin-resource-policy", b"same-origin"),
                ]
                if self._hsts:
                    extra.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                for name, value in extra:
                    if name not in existing:
                        raw.append((name, value))
                message = {**message, "headers": raw}
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _split_authority(value: str) -> tuple[str, str | None]:
    """Split ``host[:port]`` into its parts, keeping a bracketed IPv6 literal whole.

    Naively splitting on the first colon turns ``[::1]:8077`` into ``[``, which
    is why the brackets have to be respected before the port separator.
    """
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return value, None
        rest = value[end + 1 :]
        return value[: end + 1], rest[1:] if rest.startswith(":") else None
    name, separator, port = value.partition(":")
    return name, port if separator else None


def _split_origin(value: str) -> tuple[str, str, str | None]:
    """Split an Origin into ``(scheme, host, port)``; port and scheme may be absent."""
    scheme, separator, rest = value.partition("://")
    if not separator:
        # Not a URL: "null", or a bare host pattern such as *.example.com.
        return "", value, None
    host, port = _split_authority(rest)
    return scheme, host, port


#: Ports a proxy may append to a Host header without changing which authority
#: it names: the default for http and for https. A bare pattern admits them.
_DEFAULT_PORTS = frozenset({"80", "443"})


def _host_matches(host: str, patterns: tuple[str, ...]) -> bool:
    """Match a Host header against an allow-list.

    Supported patterns: an exact ``host`` or ``host:port``, ``host:*`` for any
    port on a host, ``*.example.com`` for a subdomain (on any port, or on a
    named one as ``*.example.com:8077``), and ``*`` to accept any value. A
    missing Host header is always a rejection.

    A bare ``host`` also matches ``host:80`` and ``host:443``: those name the
    same authority, and a reverse proxy is entitled to forward the port it
    received. Only a pattern that names a port is that exact about it.

    The value is sanitised first. A legitimate Host header never contains ``@``
    (that is URL userinfo, not part of the authority) or a control character,
    but a prefix-matching wildcard like ``127.0.0.1:*`` would otherwise accept
    ``127.0.0.1:8077@evil.com``. That is not reachable from a browser - a URL of
    that shape sends ``Host: evil.com`` - but a filter whose whole job is to be
    exact should be exact.
    """
    if not host:
        return False
    value = host.strip().lower()
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value) or "@" in value:
        logger.warning("rejected malformed Host header")
        return False

    name, port = _split_authority(value)
    for pattern in patterns:
        raw = pattern.strip().lower()
        pattern_name, pattern_port = _split_authority(raw)
        if pattern_name == "*":
            # Any host; a named port narrows it without excluding the default.
            if pattern_port in {None, "", "*"} or pattern_port == port:
                return True
            continue
        if pattern_name.startswith("*."):
            # A subdomain wildcard may pin a port, or leave it open. This has to
            # be tested before the "any port" branch below, which would otherwise
            # claim "*.example.com:*" and compare it as a literal name.
            if pattern_port not in {None, "", "*"} and pattern_port != port:
                continue
            # The apex is not a subdomain: *.example.com must not admit
            # example.com, and must not admit evilexample.com either.
            suffix = pattern_name[1:]
            if name.endswith(suffix) and name != suffix[1:]:
                return True
            continue
        if pattern_port == "*":
            # Any port on this exact name; the port-less form is that too.
            if name == pattern_name:
                return True
            continue
        if value == raw:
            return True
        if pattern_port is None and name == pattern_name and port in _DEFAULT_PORTS:
            # A reverse proxy legitimately forwards the port it received, and
            # that includes the scheme's default: `Host: example.com:443` names
            # the same authority as `example.com`. An operator who writes a bare
            # host means that host, not "that host only when the proxy happens to
            # drop the port" - and the failure it caused was invisible, because
            # the tool call that produced the link went over the internal network
            # while the browser's request for the image did not.
            return True
    return False


def _origin_matches(origin: str, patterns: tuple[str, ...]) -> bool:
    """Match an Origin header. An empty allow-list rejects every origin.

    The same pattern grammar as :func:`_host_matches`, with an optional scheme:
    ``http://127.0.0.1:*``, ``https://*.example.com``, or ``*``.
    """
    value = origin.strip().lower().rstrip("/")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        logger.warning("rejected malformed Origin header")
        return False
    scheme, host, port = _split_origin(value)
    for pattern in patterns:
        raw = pattern.strip().lower().rstrip("/")
        pattern_scheme, pattern_host, pattern_port = _split_origin(raw)
        if pattern_scheme and pattern_scheme != scheme:
            continue
        if pattern_host == "*":
            if pattern_port in {None, "", "*"} or pattern_port == port:
                return True
            continue
        if pattern_host.startswith("*."):
            # Subdomain wildcard, e.g. *.example.com for cdn.example.com.
            if pattern_port not in {None, "", "*"} and pattern_port != port:
                continue
            suffix = pattern_host[1:]
            if host.endswith(suffix) and host != suffix[1:]:
                return True
            continue
        if pattern_port == "*":
            if host == pattern_host:
                return True
            continue
        if value == raw:
            return True
    return False


class HostOriginGuardMiddleware:
    """Rejects requests whose Host or Origin header is not allowed.

    This is the DNS-rebinding defence the MCP specification requires for a
    server reachable from a browser, applied here to the **entire** application
    rather than only to the MCP endpoint.

    That distinction matters: the SDK validates the transport it owns, but any
    extra route added to the same application (a health check, a file server)
    sits outside that check. A browser tricked into resolving
    ``attacker.example`` to this server's address would otherwise be able to
    reach those routes. Guarding the whole app removes that gap.

    Non-browser MCP clients send no ``Origin`` header, which is always
    permitted; only a *present* Origin is checked.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_hosts: tuple[str, ...],
        allowed_origins: tuple[str, ...],
        enabled: bool = True,
    ) -> None:
        self.app = app
        self._allowed_hosts = allowed_hosts
        self._allowed_origins = allowed_origins
        self._enabled = enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._enabled or scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        host = headers.get("host")
        if not _host_matches(host or "", self._allowed_hosts):
            await self._reject(send, "host", host or "<missing>")
            return

        origin = headers.get("origin")
        if origin and not _origin_matches(origin, self._allowed_origins):
            await self._reject(send, "origin", origin)
            return

        await self.app(scope, receive, send)

    async def _reject(self, send: Send, kind: str, value: str) -> None:
        logger.warning("rejected request: %s header %r is not allowed", kind, value)
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": f"Request rejected: {kind} header is not allowed",
                },
                "id": None,
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# --------------------------------------------------------------------- URLs

#: Bytes of HMAC kept in an output token. 128 bits is far past the point where
#: guessing one is a strategy, and the token has to survive being copied.
_TOKEN_BYTES = 16


def _token_signature(secret: str, relative_path: str, expires: int) -> str:
    payload = f"{relative_path}:{expires}".encode()
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).digest()[:_TOKEN_BYTES]
    # base64url rather than hex: 22 characters instead of 32, still URL-safe,
    # and no character that markdown or HTML wants to escape.
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def build_signed_path(secret: str, relative_path: str, ttl_seconds: int, *, now: float | None = None) -> str:
    """Return ``<token>/<url-encoded path>``, where the token is ``<expiry>.<signature>``.

    **There is no query string, deliberately.** A signed URL is copied out of a
    tool result by a language model and pasted into a reply, so it has to survive
    a hostile trip: a model that drops or truncates what it reads as noisy
    parameters, markdown or HTML escaping that turns ``&`` into ``&amp;``, and a
    reverse proxy that rewrites or strips queries. Putting the expiry and the
    signature in the path removes all three failure modes at once - a model that
    keeps the path keeps the credential, and there is no ``&`` to escape.

    The path is percent-encoded as a single segment so a crafted relative path
    cannot escape the serving prefix.
    """
    expires = int(time.time() if now is None else now) + ttl_seconds
    return f"{expires}.{_token_signature(secret, relative_path, expires)}/{quote(relative_path, safe='/')}"


def signed_path_problem(
    secret: str,
    token: str,
    relative_path: str,
    *,
    now: float | None = None,
) -> str | None:
    """Why this token does not authorise this path, or ``None`` if it does.

    Returns prose rather than a boolean so the caller can put the reason in a
    log: the browser only ever shows "image unavailable", and every one of these
    cases has a different fix.
    """
    if not secret:
        return "serving is enabled without a signing secret, which cannot authorise a link"
    if not token:
        return (
            "the URL has no token segment after /files/, which is what an old-style link looks "
            "like when the query string was dropped or stripped"
        )
    expires_text, separator, signature = token.partition(".")
    if not separator or not signature:
        return "the token is malformed (expected '<expiry>.<signature>')"
    if not expires_text.isdigit():
        return "the token's expiry is not a number"
    expires = int(expires_text)
    if expires < int(time.time() if now is None else now):
        return "the link has expired"
    expected = _token_signature(secret, relative_path, expires)
    if not secrets.compare_digest(expected, signature):
        return (
            "the token does not match this file, which is what a link that was retyped, "
            "truncated or otherwise edited in transit looks like"
        )
    return None


def verify_signed_path(secret: str, token: str, relative_path: str, *, now: float | None = None) -> bool:
    """Constant-time verification of a signed output URL."""
    return signed_path_problem(secret, token, relative_path, now=now) is None


__all__ = [
    "BearerAuthMiddleware",
    "HostOriginGuardMiddleware",
    "SecurityHeadersMiddleware",
    "build_signed_path",
    "signed_path_problem",
    "verify_signed_path",
]

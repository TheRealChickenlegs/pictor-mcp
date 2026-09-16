"""SSRF-guarded remote image fetching.

Fetching a URL on behalf of a model turns the server into a proxy into whatever
network it can reach. On a Docker host that means the cloud metadata service
(``169.254.169.254``), the Docker socket's HTTP neighbours, internal admin
panels, and every service on the LAN. This module is written on the assumption
that the URL is hostile.

Defences, in order:

1. **Scheme and shape** - only ``http``/``https``, no embedded credentials, no
   non-allow-listed port.
2. **Address validation** - every address the name resolves to must be globally
   routable. A name resolving to *any* private, loopback, link-local, CGNAT,
   reserved or multicast address is refused outright, so a round-robin DNS
   answer cannot smuggle one in.
3. **IP pinning** - the connection is made to the validated address, not to the
   name. A second resolution cannot happen, so DNS rebinding (validate a public
   IP, connect to a private one) is structurally impossible rather than merely
   unlikely. TLS still uses the hostname for SNI and certificate verification.
4. **Streaming size cap** - the body is bounded while it is read, so a hostile
   server cannot exhaust memory by lying about ``Content-Length`` or by
   streaming forever.
5. **Manual redirects** - each hop is re-validated from step 1. Automatic
   redirect following is exactly how SSRF filters are bypassed.
6. **No proxy inheritance** - the connection is direct, so ``HTTP_PROXY``
   environment variables cannot silently re-route the request.

The feature is off unless ``PICTOR_ALLOW_NET_FETCH=true``.
"""

from __future__ import annotations

import contextlib
import ipaddress
import socket
import ssl
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

from ..config import FetchPolicy
from ..errors import NetworkBlockedError, NetworkDisabledError

_MAX_HEADER_BYTES = 64 * 1024
_MAX_REDIRECTS_HARD_CAP = 10
_USER_AGENT = "pictor-mcp/1.0 (+https://github.com/TheRealChickenlegs/pictor-mcp)"


@dataclass(frozen=True, slots=True)
class FetchResult:
    data: bytes
    content_type: str
    final_url: str


#: Address ranges that are never reachable as "the public internet".
#:
#: Listed explicitly rather than relying only on ``ipaddress``'s predicate
#: properties, because those properties change between Python releases: on
#: CPython 3.14, ``IPv4Address("100.64.0.1").is_private`` is ``False`` even
#: though 100.64.0.0/10 is carrier-grade NAT space that is not routable on the
#: public internet and is often used for internal services. An SSRF guard that
#: depends on version-specific classification is not a guard.
_BLOCKED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = tuple(
    ipaddress.ip_network(entry)
    for entry in (
        # IPv4 special-purpose ranges (RFC 6890 and friends)
        "0.0.0.0/8",  # "this network"
        "10.0.0.0/8",  # private
        "100.64.0.0/10",  # carrier-grade NAT / shared address space
        "127.0.0.0/8",  # loopback
        "169.254.0.0/16",  # link-local, incl. cloud metadata at .169.254
        "172.16.0.0/12",  # private
        "192.0.0.0/24",  # IETF protocol assignments
        "192.0.2.0/24",  # TEST-NET-1
        "192.88.99.0/24",  # 6to4 relay anycast
        "192.168.0.0/16",  # private
        "198.18.0.0/15",  # benchmarking
        "198.51.100.0/24",  # TEST-NET-2
        "203.0.113.0/24",  # TEST-NET-3
        "224.0.0.0/4",  # multicast
        "240.0.0.0/4",  # reserved, incl. 255.255.255.255
        # IPv6 special-purpose ranges
        "::/128",  # unspecified
        "::1/128",  # loopback
        "64:ff9b::/96",  # NAT64
        "100::/64",  # discard-only
        "2001:db8::/32",  # documentation
        "2002::/16",  # 6to4, which can embed a private IPv4 address
        "fc00::/7",  # unique local
        "fe80::/10",  # link-local
        "ff00::/8",  # multicast
    )
)


def _is_public_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True only for addresses that are safe to reach on the public internet."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        # Treat ``::ffff:169.254.169.254`` as the IPv4 address it really is,
        # so a mapped address cannot dodge the IPv4 rules.
        return _is_public_address(ip.ipv4_mapped)

    for network in _BLOCKED_NETWORKS:
        if ip.version == network.version and ip in network:
            return False

    # The explicit list above is the authority; these predicates catch anything
    # a future range addition might miss.
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or getattr(ip, "is_site_local", False)
    )


def _host_allowed(host: str, allowed: tuple[str, ...]) -> bool:
    """Match a host against the fetch allow-list.

    An empty list means "any host, subject to the address policy". A ``*`` entry
    means the same thing explicitly - it is documented in ``.env.example`` as
    "any public host", and without this branch the pattern would match nothing
    and silently block every fetch.
    """
    if not allowed:
        return True
    host = host.lower().rstrip(".")
    for pattern in allowed:
        pattern = pattern.lower().rstrip(".")
        if pattern == "*":
            return True
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]) and host != pattern[2:]:
                return True
        elif host == pattern:
            return True
    return False


class SafeFetcher:
    """Fetches images over HTTP(S) with SSRF protections."""

    __slots__ = ("_policy", "_ssl_context")

    def __init__(self, policy: FetchPolicy) -> None:
        self._policy = policy
        self._ssl_context: ssl.SSLContext | None = None

    # ------------------------------------------------------------------ API
    def fetch(self, url: str) -> FetchResult:
        """Synchronously fetch ``url``; call via a worker thread.

        Blocking sockets are used deliberately: the connect target is derived
        from a validated address, and an explicit ``socket`` gives precise
        control over exactly which peer is contacted.
        """
        policy = self._policy
        if not policy.enabled:
            raise NetworkDisabledError(
                "fetching images by URL is disabled; set PICTOR_ALLOW_NET_FETCH=true to enable it"
            )

        current = url.strip()
        max_redirects = min(policy.max_redirects, _MAX_REDIRECTS_HARD_CAP)
        for _hop in range(max_redirects + 1):
            scheme, host, port, target = self._parse_and_validate(current)
            addresses = self._resolve(host, port)
            response = self._request(scheme, host, port, target, addresses)

            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise NetworkBlockedError("redirect response had no Location header")
                current = urljoin(current, location)
                continue

            if response.status != 200:
                raise NetworkBlockedError(f"remote server returned HTTP {response.status}")

            if not response.body:
                raise NetworkBlockedError("remote server returned an empty body")
            return FetchResult(
                data=response.body,
                content_type=response.headers.get("content-type", ""),
                final_url=current,
            )

        raise NetworkBlockedError(f"too many redirects (limit {max_redirects})")

    # ------------------------------------------------------------- internals
    def _parse_and_validate(self, url: str) -> tuple[str, str, int, str]:
        try:
            parts = urlsplit(url)
        except ValueError as exc:
            raise NetworkBlockedError("malformed URL") from exc

        if parts.scheme not in {"http", "https"}:
            raise NetworkBlockedError("only http and https URLs may be fetched")
        if not parts.hostname:
            raise NetworkBlockedError("URL has no host")
        if parts.username or parts.password:
            raise NetworkBlockedError("URLs containing credentials are refused")

        host = parts.hostname
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError as exc:
            raise NetworkBlockedError("URL has an invalid port") from exc

        if port not in self._policy.allowed_ports:
            raise NetworkBlockedError(
                f"port {port} is not allowed (allowed: {', '.join(map(str, self._policy.allowed_ports))})"
            )
        if not _host_allowed(host, self._policy.allowed_hosts):
            raise NetworkBlockedError("host is not in PICTOR_FETCH_ALLOWED_HOSTS")

        # Refuse surprising encodings that could desynchronise a downstream parser.
        if any(ch in url for ch in ("\r", "\n", "\x00")):
            raise NetworkBlockedError("URL contains control characters")

        target = parts.path or "/"
        if parts.query:
            target = f"{target}?{parts.query}"
        return parts.scheme, host, port, target

    def _resolve(self, host: str, port: int) -> list[tuple[int, tuple]]:
        """Resolve and validate every address for ``host``.

        The *whole* answer set must be publicly routable. Rejecting the request
        when any address is internal removes the round-robin rebinding trick,
        where a name returns one public and one private address.
        """
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise NetworkBlockedError("host could not be resolved") from exc
        if not infos:
            raise NetworkBlockedError("host resolved to no addresses")

        validated: list[tuple[int, tuple]] = []
        for family, _type, _proto, _canon, sockaddr in infos:
            literal = sockaddr[0]
            try:
                ip = ipaddress.ip_address(literal.split("%", 1)[0])
            except ValueError as exc:
                raise NetworkBlockedError("host resolved to an unparseable address") from exc
            if not _is_public_address(ip):
                # The hint is the whole point of this message. Told only
                # "private address", a caller reads a transient network problem
                # and retries the same container-to-container URL; an image on
                # this network is not reachable by URL at all.
                raise NetworkBlockedError(
                    "host resolves to a private, loopback, link-local or reserved address; only "
                    "publicly routable URLs can be fetched, so an image on this network has to be "
                    "mounted under PICTOR_INPUT_ROOTS and passed as `path` instead"
                )
            validated.append((family, sockaddr))
        return validated

    def _request(
        self,
        scheme: str,
        host: str,
        port: int,
        target: str,
        addresses: list[tuple[int, tuple]],
    ) -> _RawResponse:
        timeout = self._policy.timeout_seconds
        last_error: OSError | None = None

        for family, sockaddr in addresses:
            sock: socket.socket | None = None
            try:
                # Connect to the *validated literal address*, never the name.
                sock = socket.socket(family, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect(sockaddr)

                if scheme == "https":
                    context = self._get_ssl_context()
                    # SNI and certificate verification use the hostname, so
                    # pinning the IP costs nothing in TLS correctness.
                    sock = context.wrap_socket(sock, server_hostname=host)

                sock.sendall(self._build_request(host, port, target, scheme))
                return self._read_response(sock)
            except OSError as exc:
                last_error = exc
                continue
            finally:
                # Runs after `_read_response` completes on the success path, so
                # the descriptor is always released exactly once.
                if sock is not None:
                    with contextlib.suppress(OSError):  # pragma: no cover - best effort
                        sock.close()

        raise NetworkBlockedError(f"could not connect to host: {last_error or 'connection failed'}")

    def _build_request(self, host: str, port: int, target: str, scheme: str) -> bytes:
        default_port = 443 if scheme == "https" else 80
        host_header = host if port == default_port else f"{host}:{port}"
        # Every header value is validated upstream, so no CRLF injection is
        # possible into the request line or headers.
        lines = [
            f"GET {target} HTTP/1.1",
            f"Host: {host_header}",
            f"User-Agent: {_USER_AGENT}",
            "Accept: image/*,application/octet-stream;q=0.5",
            # No compression: it removes an entire decoder from the trust path
            # and stops a small compressed body from expanding without bound.
            "Accept-Encoding: identity",
            "Connection: close",
            "",
            "",
        ]
        return "\r\n".join(lines).encode("latin-1")

    def _get_ssl_context(self) -> ssl.SSLContext:
        if self._ssl_context is not None:
            return self._ssl_context
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        if not self._policy.verify_tls:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        self._ssl_context = context
        return context

    def _read_response(self, sock: socket.socket) -> _RawResponse:
        reader = _SocketReader(sock, self._policy.max_bytes)
        status_line = reader.read_line(_MAX_HEADER_BYTES)
        parts = status_line.split(" ", 2)
        if len(parts) < 2 or not parts[0].startswith("HTTP/"):
            raise NetworkBlockedError("remote server sent a malformed status line")
        try:
            status = int(parts[1])
        except ValueError as exc:
            raise NetworkBlockedError("remote server sent a malformed status code") from exc

        headers: dict[str, str] = {}
        while True:
            line = reader.read_line(_MAX_HEADER_BYTES)
            if not line:
                break
            name, sep, value = line.partition(":")
            if not sep:
                raise NetworkBlockedError("remote server sent a malformed header")
            headers.setdefault(name.strip().lower(), value.strip())

        if status in {204, 304} or status < 200:
            return _RawResponse(status=status, headers=headers, body=b"")
        if status in {301, 302, 303, 307, 308}:
            return _RawResponse(status=status, headers=headers, body=b"")

        declared = headers.get("content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError as exc:
                raise NetworkBlockedError("remote server sent an invalid Content-Length") from exc
            if length > self._policy.max_bytes:
                raise NetworkBlockedError(
                    f"remote image is {length} bytes, exceeding the {self._policy.max_bytes} byte fetch limit"
                )

        transfer_encoding = headers.get("transfer-encoding", "").lower()
        if "chunked" in transfer_encoding:
            body = reader.read_chunked()
        elif declared is not None:
            body = reader.read_exactly(int(declared))
        else:
            body = reader.read_to_eof()

        return _RawResponse(status=status, headers=headers, body=body)


@dataclass(frozen=True, slots=True)
class _RawResponse:
    status: int
    headers: dict[str, str]
    body: bytes


class _SocketReader:
    """Bounded reader over a blocking socket."""

    __slots__ = ("_buffer", "_limit", "_sock")

    def __init__(self, sock: socket.socket, limit: int) -> None:
        self._sock = sock
        self._buffer = bytearray()
        self._limit = limit

    def _fill(self) -> bool:
        try:
            chunk = self._sock.recv(64 * 1024)
        except TimeoutError as exc:
            raise NetworkBlockedError("remote server timed out") from exc
        except OSError as exc:
            raise NetworkBlockedError("connection failed while reading the response") from exc
        if not chunk:
            return False
        self._buffer.extend(chunk)
        return True

    def read_line(self, max_bytes: int) -> str:
        while b"\r\n" not in self._buffer:
            if len(self._buffer) > max_bytes:
                raise NetworkBlockedError("remote server sent an oversized header line")
            if not self._fill():
                break
        index = self._buffer.find(b"\r\n")
        if index < 0:
            line = bytes(self._buffer)
            self._buffer.clear()
            return line.decode("latin-1")
        line = bytes(self._buffer[:index])
        del self._buffer[: index + 2]
        return line.decode("latin-1")

    def read_exactly(self, count: int) -> bytes:
        if count > self._limit:
            raise NetworkBlockedError("remote image exceeds the fetch size limit")
        while len(self._buffer) < count:
            if not self._fill():
                raise NetworkBlockedError("remote server closed the connection early")
        data = bytes(self._buffer[:count])
        del self._buffer[:count]
        return data

    def read_to_eof(self) -> bytes:
        while len(self._buffer) <= self._limit:
            if not self._fill():
                break
        if len(self._buffer) > self._limit:
            raise NetworkBlockedError("remote image exceeds the fetch size limit")
        data = bytes(self._buffer)
        self._buffer.clear()
        return data

    def read_chunked(self) -> bytes:
        body = bytearray()
        while True:
            size_line = self.read_line(1024).split(";", 1)[0].strip()
            if not size_line:
                raise NetworkBlockedError("remote server sent a malformed chunk header")
            try:
                size = int(size_line, 16)
            except ValueError as exc:
                raise NetworkBlockedError("remote server sent an invalid chunk size") from exc
            if size == 0:
                # Consume trailers up to the terminating blank line.
                while self.read_line(_MAX_HEADER_BYTES):
                    pass
                break
            if len(body) + size > self._limit:
                raise NetworkBlockedError("remote image exceeds the fetch size limit")
            body.extend(self.read_exactly(size))
            if self.read_line(2):
                raise NetworkBlockedError("remote server sent a malformed chunk terminator")
        return bytes(body)


__all__ = ["FetchResult", "SafeFetcher"]

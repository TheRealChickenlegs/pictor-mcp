"""SSRF guard tests.

The guard is exercised in two layers:

* the **policy** (address classification, scheme/port/host allow-lists) is tested
  directly, because those are pure functions and the interesting cases are
  addresses no test machine can route to;
* the **HTTP client** (redirects, chunked bodies, size caps) is tested against a
  real local server, with only the address classifier stubbed - so the parsing
  and bounding logic runs for real rather than against a mock.
"""

from __future__ import annotations

import http.server
import ipaddress
import socket
import threading
from collections.abc import Iterator

import pytest

from pictor_mcp.config import FetchPolicy
from pictor_mcp.errors import NetworkBlockedError, NetworkDisabledError
from pictor_mcp.security import net
from pictor_mcp.security.net import SafeFetcher, _is_public_address


class TestAddressClassification:
    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",
            "127.1.2.3",
            "10.0.0.1",
            "172.16.5.4",
            "192.168.1.1",
            "169.254.169.254",  # cloud metadata: the classic SSRF target
            "169.254.0.1",
            "0.0.0.0",
            "100.64.0.1",  # CGNAT
            "192.0.2.1",  # TEST-NET-1, not globally routable
            "198.18.0.1",  # benchmarking range
            "224.0.0.1",  # multicast
            "255.255.255.255",
            "::1",
            "fe80::1",
            "fc00::1",
            "::ffff:127.0.0.1",  # IPv4-mapped loopback
            "::ffff:169.254.169.254",
            "ff02::1",
        ],
    )
    def test_internal_addresses_are_refused(self, address: str) -> None:
        assert _is_public_address(ipaddress.ip_address(address)) is False

    @pytest.mark.parametrize("address", ["1.1.1.1", "8.8.8.8", "93.184.216.34", "2606:4700::1111"])
    def test_public_addresses_are_allowed(self, address: str) -> None:
        assert _is_public_address(ipaddress.ip_address(address)) is True


class TestUrlValidation:
    def _fetcher(self, **overrides: object) -> SafeFetcher:
        policy = FetchPolicy(enabled=True, **overrides)  # type: ignore[arg-type]
        return SafeFetcher(policy)

    def test_disabled_by_default(self) -> None:
        with pytest.raises(NetworkDisabledError):
            SafeFetcher(FetchPolicy()).fetch("http://example.com/a.png")

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "gopher://example.com/",
            "ftp://example.com/x.png",
            "data:image/png;base64,AAAA",
            "javascript:alert(1)",
            "/etc/passwd",
            "http://",
        ],
    )
    def test_rejects_non_http_schemes(self, url: str) -> None:
        with pytest.raises(NetworkBlockedError):
            self._fetcher().fetch(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://user:pass@example.com/x.png",
            "http://user@example.com/x.png",
        ],
    )
    def test_rejects_embedded_credentials(self, url: str) -> None:
        with pytest.raises(NetworkBlockedError):
            self._fetcher().fetch(url)

    def test_rejects_a_port_outside_the_allow_list(self) -> None:
        with pytest.raises(NetworkBlockedError):
            self._fetcher().fetch("http://example.com:8080/x.png")

    def test_allows_an_explicitly_permitted_port(self) -> None:
        scheme, host, port, _ = self._fetcher(allowed_ports=(80, 443, 8080))._parse_and_validate(
            "http://example.com:8080/x.png"
        )
        assert (scheme, host, port) == ("http", "example.com", 8080)

    def test_host_allow_list_is_honoured(self) -> None:
        fetcher = self._fetcher(allowed_hosts=("images.example.com",))
        fetcher._parse_and_validate("https://images.example.com/a.png")
        with pytest.raises(NetworkBlockedError):
            fetcher._parse_and_validate("https://elsewhere.example.com/a.png")

    def test_host_allow_list_supports_wildcards(self) -> None:
        fetcher = self._fetcher(allowed_hosts=("*.example.com",))
        fetcher._parse_and_validate("https://cdn.example.com/a.png")
        # The bare apex is not covered by a subdomain wildcard.
        with pytest.raises(NetworkBlockedError):
            fetcher._parse_and_validate("https://example.com/a.png")

    @pytest.mark.parametrize("url", ["http://example.com/a\r\nX-Evil: 1", "http://exa\x00mple.com/a"])
    def test_rejects_header_injection_attempts(self, url: str) -> None:
        with pytest.raises(NetworkBlockedError):
            self._fetcher()._parse_and_validate(url)


class TestResolutionPolicy:
    """A name resolving to *any* internal address must be refused wholesale."""

    def test_rejects_a_name_resolving_to_a_private_address(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 80))],
        )
        with pytest.raises(NetworkBlockedError) as caught:
            SafeFetcher(FetchPolicy(enabled=True))._resolve("internal.example", 80)
        # The caller is a model, and it will otherwise retry the same
        # container-to-container URL: "private address" reads as a transient
        # network fault. The message has to name the route that does work.
        assert "path" in str(caught.value), caught.value
        assert "PICTOR_INPUT_ROOTS" in str(caught.value), caught.value

    def test_rejects_a_mixed_public_private_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Round-robin DNS must not be able to smuggle in a private address."""
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *a, **k: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80)),
            ],
        )
        with pytest.raises(NetworkBlockedError):
            SafeFetcher(FetchPolicy(enabled=True))._resolve("mixed.example", 80)

    def test_accepts_an_all_public_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *a, **k: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443)),
            ],
        )
        assert len(SafeFetcher(FetchPolicy(enabled=True))._resolve("good.example", 443)) == 2

    def test_unresolvable_host_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise socket.gaierror("nope")

        monkeypatch.setattr(socket, "getaddrinfo", boom)
        with pytest.raises(NetworkBlockedError):
            SafeFetcher(FetchPolicy(enabled=True))._resolve("nowhere.invalid", 80)


# --------------------------------------------------------------------- server


class _Handler(http.server.BaseHTTPRequestHandler):
    """A tiny local origin server with routes chosen by path."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: object) -> None:  # keep test output clean
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/image.png":
            body = b"\x89PNG\r\n\x1a\n" + b"A" * 64
            self._respond(200, body, "image/png")
        elif path == "/huge":
            self._respond(200, b"B" * (5 * 1024 * 1024), "image/png")
        elif path == "/chunked":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in (b"X" * 16, b"Y" * 16, b"Z" * 16):
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        elif path == "/redirect":
            self._respond(302, b"", "text/plain", location="/image.png")
        elif path == "/redirect-loop":
            self._respond(302, b"", "text/plain", location="/redirect-loop")
        elif path == "/redirect-private":
            self._respond(302, b"", "text/plain", location="http://169.254.169.254/latest/meta-data/")
        elif path == "/nolen":
            # Declares no length and no chunking, then closes.
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(b"C" * 32)
        elif path == "/error":
            self._respond(500, b"boom", "text/plain")
        else:
            self._respond(404, b"", "text/plain")

    def _respond(self, status: int, body: bytes, content_type: str, location: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if location:
            self.send_header("Location", location)
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def origin(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A real local HTTP server, with the public-address policy relaxed.

    Only :func:`_is_public_address` is stubbed. Everything else - request
    building, status parsing, redirect handling, chunk decoding and size
    bounding - runs against a real socket.
    """

    class _QuietServer(http.server.ThreadingHTTPServer):
        # The size-cap tests make the client hang up mid-body, which is the
        # behaviour under test; the server's default traceback would only add
        # noise.
        def handle_error(self, request, client_address):
            pass

    server = _QuietServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    monkeypatch.setattr(net, "_is_public_address", lambda _ip: True)
    # The fetcher connects to the literal address, so allow the loopback port.
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _fetcher_for(origin: str, **overrides: object) -> SafeFetcher:
    port = int(origin.rsplit(":", 1)[1])
    defaults: dict[str, object] = {"allowed_ports": (port,)}
    defaults.update(overrides)
    return SafeFetcher(FetchPolicy(enabled=True, **defaults))  # type: ignore[arg-type]


class TestHttpClient:
    def test_fetches_an_image(self, origin: str) -> None:
        result = _fetcher_for(origin).fetch(f"{origin}/image.png")
        assert result.data.startswith(b"\x89PNG")
        assert result.content_type == "image/png"

    def test_follows_a_redirect(self, origin: str) -> None:
        assert _fetcher_for(origin).fetch(f"{origin}/redirect").data.startswith(b"\x89PNG")

    def test_refuses_a_redirect_to_a_private_address(self, origin: str) -> None:
        """Each hop is re-validated, so an open redirect cannot reach metadata."""
        monkeypatch_target = net._is_public_address
        try:
            net._is_public_address = lambda ip: ip != ipaddress.ip_address("169.254.169.254")
            with pytest.raises(NetworkBlockedError):
                _fetcher_for(origin).fetch(f"{origin}/redirect-private")
        finally:
            net._is_public_address = monkeypatch_target

    def test_caps_redirect_depth(self, origin: str) -> None:
        with pytest.raises(NetworkBlockedError):
            _fetcher_for(origin, max_redirects=2).fetch(f"{origin}/redirect-loop")

    def test_enforces_the_size_cap_on_a_declared_length(self, origin: str) -> None:
        with pytest.raises(NetworkBlockedError):
            _fetcher_for(origin, max_bytes=4096).fetch(f"{origin}/huge")

    def test_enforces_the_size_cap_on_a_chunked_body(self, origin: str) -> None:
        with pytest.raises(NetworkBlockedError):
            _fetcher_for(origin, max_bytes=8).fetch(f"{origin}/chunked")

    def test_decodes_a_chunked_body(self, origin: str) -> None:
        assert _fetcher_for(origin).fetch(f"{origin}/chunked").data == b"X" * 16 + b"Y" * 16 + b"Z" * 16

    def test_reads_a_body_with_no_length(self, origin: str) -> None:
        assert len(_fetcher_for(origin).fetch(f"{origin}/nolen").data) == 32

    def test_reports_a_non_200_status(self, origin: str) -> None:
        with pytest.raises(NetworkBlockedError):
            _fetcher_for(origin).fetch(f"{origin}/error")

    def test_reports_a_missing_resource(self, origin: str) -> None:
        with pytest.raises(NetworkBlockedError):
            _fetcher_for(origin).fetch(f"{origin}/missing")

    def test_sends_no_compression_request(self, origin: str) -> None:
        """Accept-Encoding: identity keeps a decompressor out of the trust path."""
        request = _fetcher_for(origin)._build_request("example.com", 80, "/a", "http")
        assert b"Accept-Encoding: identity" in request
        assert b"Connection: close" in request

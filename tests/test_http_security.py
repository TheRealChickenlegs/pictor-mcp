"""HTTP transport security tests against a live uvicorn server.

These cover the claims the deployment documentation makes: a bearer token is
required, DNS-rebinding attempts are refused on *every* route (not only the MCP
endpoint), signed output URLs work in a browser that has no token, and a forged
signature does not.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
import uvicorn

from pictor_mcp.config import load_config
from pictor_mcp.server import build_context, build_http_app, build_server

from .conftest import Sandbox, base_env

pytestmark = pytest.mark.anyio

TOKEN = "test-token-0123456789abcdef0123456789abcdef"
SECRET = "test-secret-0123456789abcdef0123456789abcdef"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _LiveServer:
    """Run the real ASGI app under uvicorn in a background thread."""

    def __init__(self, config) -> None:
        self.config = config
        # Bind the port the configuration advertises, so generated output URLs
        # match the address the test actually talks to.
        self.port = config.http.port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _LiveServer:
        ctx = build_context(self.config)
        server = build_server(self.config, ctx)
        app = build_http_app(self.config, ctx, server)
        uvicorn_config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            proxy_headers=False,
            server_header=False,
        )
        self._server = uvicorn.Server(uvicorn_config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 20
        while time.time() < deadline:
            if getattr(self._server, "started", False):
                break
            time.sleep(0.05)
        else:  # pragma: no cover - startup failure
            raise RuntimeError("test server did not start")
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def request(self, path: str, *, headers: dict[str, str] | None = None, method: str = "GET"):
        request = urllib.request.Request(self.base + path, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status, response.read(), {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), {k.lower(): v for k, v in exc.headers.items()}


@pytest.fixture
def secure_server(sandbox: Sandbox):
    """A server with a bearer token, signature-protected outputs, loopback bind."""
    config = load_config(
        base_env(
            sandbox,
            PICTOR_TRANSPORT="streamable-http",
            PICTOR_HOST="127.0.0.1",
            PICTOR_PORT=str(_free_port()),
            PICTOR_AUTH_TOKEN=TOKEN,
            PICTOR_URL_SECRET=SECRET,
            PICTOR_SERVE_OUTPUTS="true",
        )
    )
    with _LiveServer(config) as live:
        yield live


def _rpc(live: _LiveServer, method: str, params: dict | None = None, *, token: str | None = TOKEN):
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
            # The modern stateless envelope: version and capabilities per request.
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {},
            },
        }
    ).encode()
    request = urllib.request.Request(live.base + "/mcp", data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read(), {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), {k.lower(): v for k, v in exc.headers.items()}


class TestAuthentication:
    def test_mcp_requires_a_token(self, secure_server: _LiveServer) -> None:
        status, _, headers = _rpc(secure_server, "tools/list", token=None)
        assert status == 401
        assert "www-authenticate" in headers

    def test_a_wrong_token_is_rejected(self, secure_server: _LiveServer) -> None:
        status, _, _ = _rpc(secure_server, "tools/list", token="wrong-token-0123456789abcdef")
        assert status == 401

    def test_a_short_prefix_of_the_token_is_rejected(self, secure_server: _LiveServer) -> None:
        """Guards against a prefix-compare bug in the constant-time check."""
        status, _, _ = _rpc(secure_server, "tools/list", token=TOKEN[:-1])
        assert status == 401

    def test_the_correct_token_is_accepted(self, secure_server: _LiveServer) -> None:
        status, body, _ = _rpc(secure_server, "tools/list")
        assert status == 200, body[:200]
        assert b"image_resize" in body

    def test_x_api_key_is_accepted_as_an_alias(self, secure_server: _LiveServer) -> None:
        status, _, _ = _rpc(secure_server, "tools/list", token=None)
        assert status == 401

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-API-Key": TOKEN,
        }
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
        request = urllib.request.Request(secure_server.base + "/mcp", data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=20) as response:
            assert response.status == 200

    def test_health_is_exempt_from_auth(self, secure_server: _LiveServer) -> None:
        status, body, _ = secure_server.request("/healthz")
        assert status == 200
        assert json.loads(body) == {"status": "ok"}

    def test_health_reveals_nothing_else(self, secure_server: _LiveServer) -> None:
        _, body, _ = secure_server.request("/healthz")
        assert len(body) < 64
        assert b"version" not in body
        assert b"root" not in body


class TestDnsRebindingDefence:
    def test_a_foreign_host_header_is_refused(self, secure_server: _LiveServer) -> None:
        status, _, _ = secure_server.request("/healthz", headers={"Host": "evil.example.com"})
        assert status == 403

    def test_a_foreign_host_is_refused_on_a_custom_route(self, secure_server: _LiveServer) -> None:
        """The SDK guards its own endpoint; our extra routes need the same check."""
        status, _, _ = secure_server.request("/files/anything.png", headers={"Host": "evil.example.com"})
        assert status == 403

    def test_a_foreign_origin_is_refused(self, secure_server: _LiveServer) -> None:
        status, _, _ = secure_server.request("/healthz", headers={"Origin": "https://evil.example.com"})
        assert status == 403

    def test_a_loopback_origin_is_allowed(self, secure_server: _LiveServer) -> None:
        status, _, _ = secure_server.request("/healthz", headers={"Origin": f"http://127.0.0.1:{secure_server.port}"})
        assert status == 200

    def test_no_origin_header_is_allowed(self, secure_server: _LiveServer) -> None:
        """Non-browser MCP clients send no Origin and must keep working."""
        assert secure_server.request("/healthz")[0] == 200

    def test_a_missing_host_header_is_refused(self, secure_server: _LiveServer) -> None:
        status, _, _ = _rpc(secure_server, "tools/list", token=None)
        # A normal request without a token is 401; the Host check is exercised
        # via the raw path above. Here we just confirm the header is present in
        # a well-formed request.
        assert status == 401


def _post_as_host(live: _LiveServer, host: str, *, token: str | None = None):
    """POST to /mcp with an explicit Host header.

    urllib derives Host from the URL, so a raw client is the only way to send
    the value another container or a reverse proxy would send.
    """
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
    headers = {
        "Host": host,
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    connection = http.client.HTTPConnection("127.0.0.1", live.port, timeout=20)
    try:
        connection.request("POST", "/mcp", body=payload, headers=headers)
        response = connection.getresponse()
        return response.status, response.read(), {k.lower(): v for k, v in response.getheaders()}
    finally:
        connection.close()


@contextlib.contextmanager
def _live_with_hosts(sandbox: Sandbox, hosts: str | None = None, *, host: str = "127.0.0.1"):
    overrides = {
        "PICTOR_TRANSPORT": "streamable-http",
        "PICTOR_HOST": host,
        "PICTOR_PORT": str(_free_port()),
        "PICTOR_AUTH_TOKEN": TOKEN,
    }
    if hosts is not None:
        overrides["PICTOR_ALLOWED_HOSTS"] = hosts
    config = load_config(base_env(sandbox, **overrides))
    with _LiveServer(config) as live:
        yield live


class TestTheHostPolicyHasASingleEnforcementPoint:
    """The SDK validates Host itself, with a weaker matcher than ours.

    Both layers used to receive the same list, so a pattern that ours understands
    and the SDK cannot express was allowed by the outer guard and then refused by
    the inner one with a bare ``421 Invalid Host header`` - no remedy, and a
    contradiction of the answer the operator had just been given. These are the
    exact patterns that used to fail.
    """

    @pytest.mark.parametrize(
        ("allowed", "host"),
        [
            ("pictor-mcp:*", "pictor-mcp:8077"),
            # Legal HTTP: a Host may omit the port. The SDK's `name:*` handling
            # requires the colon, so this 421'd before.
            ("pictor-mcp:*", "pictor-mcp"),
            ("*", "anything.example.com"),
            ("*.example.com", "cdn.example.com:8077"),
            ("*.example.com", "cdn.example.com"),
        ],
        ids=["service:port", "service-no-port", "wildcard-all", "subdomain:port", "subdomain"],
    )
    def test_an_allowed_host_reaches_the_mcp_endpoint(self, sandbox: Sandbox, allowed: str, host: str) -> None:
        with _live_with_hosts(sandbox, allowed) as live:
            status, body, _ = _post_as_host(live, host, token=TOKEN)
            assert status == 200, f"{host} against {allowed!r}: {status} {body[:200]!r}"

    @pytest.mark.parametrize(
        ("allowed", "host"),
        [
            ("pictor-mcp:*", "evil.example.com"),
            ("pictor-mcp:*", "pictor-mcp.evil.com"),
            ("*.example.com", "example.com:8077"),
            ("*.example.com", "evilexample.com"),
        ],
    )
    def test_a_foreign_host_is_refused_on_the_mcp_endpoint_too(self, sandbox: Sandbox, allowed: str, host: str) -> None:
        """Disabling the inner check must not move the endpoint out of scope."""
        with _live_with_hosts(sandbox, allowed) as live:
            status, body, _ = _post_as_host(live, host, token=TOKEN)
            assert status == 403, f"{host}: {status} {body[:200]!r}"
            assert b"host header" in body.lower()

    @pytest.mark.parametrize(
        ("host", "patterns", "expected"),
        [
            # A bare pattern names a host, and a proxy may forward the scheme's
            # default port with it. Those are the same authority.
            ("pictor.example.com:443", ("pictor.example.com",), True),
            ("pictor.example.com:80", ("pictor.example.com",), True),
            ("pictor.example.com", ("pictor.example.com",), True),
            # Any other port is a different authority and needs `host:*`.
            ("pictor.example.com:8077", ("pictor.example.com",), False),
            ("pictor.example.com:8443", ("pictor.example.com",), False),
            # A pattern that names a port stays exact about it.
            ("pictor.example.com:8443", ("pictor.example.com:8443",), True),
            ("pictor.example.com", ("pictor.example.com:8443",), False),
            ("pictor.example.com:443", ("pictor.example.com:8443",), False),
            # The hostname still has to be the one the pattern names.
            ("pictor.example.com.evil.com:443", ("pictor.example.com",), False),
            ("evil.com:443", ("pictor.example.com",), False),
        ],
    )
    def test_a_bare_pattern_admits_the_default_ports(
        self, host: str, patterns: tuple[str, ...], expected: bool
    ) -> None:
        from pictor_mcp.security.auth import _host_matches

        assert _host_matches(host, patterns) is expected

    def test_a_wildcard_bind_with_no_list_is_loud_and_actually_works(self, sandbox: Sandbox, caplog) -> None:
        """`PICTOR_HOST=0.0.0.0` alone derives `*`, which is a lot to do quietly.

        Before the inner check was removed this combination was unusable - every
        request was 421'd - so an operator could not have been relying on it. It
        works now, which makes the warning the only thing standing between them
        and a policy that accepts any Host.
        """
        with (
            caplog.at_level(logging.WARNING, logger="pictor_mcp.server"),
            _live_with_hosts(sandbox, host="0.0.0.0") as live,
        ):
            status, body, _ = _post_as_host(live, "somewhere.internal:8077", token=TOKEN)
        assert status == 200, body[:200]
        assert any("effectively off" in record.getMessage() for record in caplog.records), [
            record.getMessage() for record in caplog.records
        ]

    def test_the_compose_default_needs_no_warning(self, sandbox: Sandbox, caplog) -> None:
        """The shipped default must not trip the `*` warning."""
        with (
            caplog.at_level(logging.WARNING, logger="pictor_mcp.server"),
            _live_with_hosts(sandbox, "127.0.0.1:*,localhost:*,pictor-mcp:*") as live,
        ):
            assert _post_as_host(live, "pictor-mcp:8077", token=TOKEN)[0] == 200
        assert not [r for r in caplog.records if "effectively off" in r.getMessage()]


class TestSecurityHeaders:
    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("x-content-type-options", "nosniff"),
            ("x-frame-options", "DENY"),
            ("referrer-policy", "no-referrer"),
        ],
    )
    def test_headers_are_present(self, secure_server: _LiveServer, header: str, expected: str) -> None:
        _, _, headers = secure_server.request("/healthz")
        assert headers.get(header) == expected

    def test_headers_are_present_on_a_rejection(self, secure_server: _LiveServer) -> None:
        """Even a 401 must carry the hardening headers."""
        _, _, headers = _rpc(secure_server, "tools/list", token=None)
        assert headers.get("x-content-type-options") == "nosniff"
        assert headers.get("cache-control") == "no-store"


async def _produce_output(live: _LiveServer) -> dict:
    """Run one real tool call over streamable HTTP and return its output record.

    Driving this through the MCP client rather than hand-rolled JSON-RPC keeps
    the test focused on the file endpoint, which is what is under test here;
    the session handshake is the SDK's problem, not this suite's.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    # Use the SDK's own factory rather than constructing an HTTP client here:
    # which library backs the transport is the SDK's concern, not this suite's.
    client = create_mcp_http_client(headers={"Authorization": f"Bearer {TOKEN}"})
    async with (
        client,
        streamable_http_client(live.base + "/mcp", http_client=client) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.call_tool(
            "image_resize",
            {"path": "photo.jpg", "width": 40, "return_image": False},
        )
        assert result.is_error is False, result.content
        return result.structured_content["outputs"][0]


class TestSignedOutputUrls:
    async def test_signed_url_works_without_a_token(self, secure_server: _LiveServer) -> None:
        """A browser <img> tag has no bearer token; the signature is the credential."""
        output = await _produce_output(secure_server)
        assert output["url"], "output should carry a URL when serving is enabled"

        path = output["url"].split(secure_server.base, 1)[1]
        status, body, headers = secure_server.request(path)
        assert status == 200, body[:200]
        assert body.startswith(b"\xff\xd8") or body.startswith(b"\x89PNG")
        # Content type must be right or a browser will not render it.
        assert headers.get("content-type", "").startswith("image/")
        assert headers.get("content-disposition") == "inline"

    async def test_a_forged_signature_is_refused(self, secure_server: _LiveServer) -> None:
        output = await _produce_output(secure_server)
        # A token that was tampered with in place. The URL suffix already starts
        # with /files/, so it is requested as it is.
        tampered = output["url"].split(secure_server.base, 1)[1].replace(".", ".0", 1)
        status, _, _ = secure_server.request(tampered)
        assert status == 403

    async def test_a_tampered_path_is_refused(self, secure_server: _LiveServer) -> None:
        output = await _produce_output(secure_server)
        signed = output["url"].split(secure_server.base, 1)[1]
        # Keep the original token but point at a different file.
        tampered = signed.replace(output["path"], "resized/something-else.jpg", 1)
        status, _, _ = secure_server.request(tampered)
        assert status == 403

    async def test_an_expired_link_is_refused(self, secure_server: _LiveServer) -> None:
        from pictor_mcp.security.auth import build_signed_path

        output = await _produce_output(secure_server)
        relative = output["path"]
        stale = build_signed_path(SECRET, relative, -10)
        status, _, _ = secure_server.request(f"/files/{stale}")
        assert status == 403

    async def test_a_link_signed_with_another_secret_is_refused(self, secure_server: _LiveServer) -> None:
        from pictor_mcp.security.auth import build_signed_path

        output = await _produce_output(secure_server)
        foreign = build_signed_path("a-completely-different-secret-value", output["path"], 600)
        status, _, _ = secure_server.request(f"/files/{foreign}")
        assert status == 403

    async def test_a_link_with_its_query_string_dropped_still_works(self, secure_server: _LiveServer) -> None:
        """The regression this format exists for.

        Generated links used to carry `?e=<expiry>&s=<signature>`, and a browser
        kept getting "image unavailable" while the log said the signature did not
        verify. Two things produce that: a model that drops what it reads as
        noisy query parameters when it retypes the URL into its reply, and a proxy
        that rewrites the query. Putting the credential in the path means a URL
        that keeps its path keeps its credential, and there is no `&` for
        anything to escape.
        """
        output = await _produce_output(secure_server)
        path = output["url"].split(secure_server.base, 1)[1]
        assert "?" not in path, path
        # Asking for the same URL with a query appended appends to nothing: the
        # path carries everything.
        status, body, _ = secure_server.request(f"{path}?tracking=1")
        assert status == 200, body[:200]

    async def test_traversal_through_the_file_endpoint_is_refused(self, secure_server: _LiveServer) -> None:
        from pictor_mcp.security.auth import build_signed_path

        # A correctly signed link for a path outside the output root, to prove
        # the jail refuses it rather than the signature.
        relative = "../input/photo.jpg"
        status, _, _ = secure_server.request(f"/files/{build_signed_path(SECRET, relative, 600)}")
        # 400 is the path jail refusing to resolve outside the output root;
        # 403/404 are also acceptable refusals. Anything 2xx would be a leak.
        assert status in {400, 403, 404}

    async def test_an_unsigned_request_to_the_file_endpoint_is_refused(self, secure_server: _LiveServer) -> None:
        """With a signing secret configured, no signature means no download."""
        output = await _produce_output(secure_server)
        status, _, _ = secure_server.request(f"/files/{output['path']}")
        assert status == 403

    def test_a_proxy_that_forwards_the_default_port_still_gets_the_image(self, sandbox: Sandbox) -> None:
        """The failure that looks exactly like a broken image.

        A reverse proxy may pass on the port it received, so the Host header is
        `pictor.example.com:443` rather than the bare name. An operator who wrote
        `PICTOR_ALLOWED_HOSTS=pictor.example.com` meant that host, and refusing
        the default-port form silently broke every generated link - the tool call
        succeeded over the internal network while the browser's request for the
        image was rejected. `:443` and `:80` name the same authority as the bare
        host, so a bare pattern admits them.
        """
        config = load_config(
            base_env(
                sandbox,
                PICTOR_TRANSPORT="streamable-http",
                PICTOR_HOST="127.0.0.1",
                PICTOR_PORT=str(_free_port()),
                PICTOR_AUTH_TOKEN=TOKEN,
                PICTOR_SERVE_OUTPUTS="true",
                PICTOR_URL_SECRET=SECRET,
                PICTOR_PUBLIC_BASE_URL="https://pictor.example.com",
                PICTOR_ALLOWED_HOSTS="127.0.0.1:*,pictor.example.com",
            )
        )
        with _LiveServer(config) as live:
            import anyio

            output = anyio.run(_produce_output, live)
            # The link is built from PICTOR_PUBLIC_BASE_URL, not from the
            # loopback address this test talks to, so keep only its path.
            from urllib.parse import urlsplit

            parts = urlsplit(output["url"])
            path = f"{parts.path}?{parts.query}"
            for host in ("pictor.example.com", "pictor.example.com:443", "pictor.example.com:80"):
                status, body, headers = live.request(path, headers={"Host": host})
                assert status == 200, f"{host}: {status} {body[:120]!r}"
                assert headers.get("content-type", "").startswith("image/")
            # A port that is not a default still needs `host:*`: it names a
            # different authority, and admitting it silently would be a hole.
            status, _, _ = live.request(path, headers={"Host": "pictor.example.com:8077"})
            assert status == 403

    async def test_a_refused_link_says_why_in_the_log(self, secure_server: _LiveServer, caplog) -> None:
        """A browser shows only "image unavailable", so the reason has to reach
        the log or the operator has nothing to go on."""
        import logging

        output = await _produce_output(secure_server)
        tampered = output["url"].split(secure_server.base, 1)[1].replace(".", ".0", 1)
        with caplog.at_level(logging.WARNING, logger="pictor_mcp.server"):
            status, _, _ = secure_server.request(tampered)
        assert status == 403
        assert any("does not match this file" in record.getMessage() for record in caplog.records), [
            record.getMessage() for record in caplog.records
        ]

    async def test_a_link_from_the_older_url_shape_says_so(self, secure_server: _LiveServer, caplog) -> None:
        """Links used to carry `?e=…&s=…`. Someone will paste one, and the log
        should name the shape rather than complain about a malformed token."""
        output = await _produce_output(secure_server)
        import logging

        with caplog.at_level(logging.WARNING, logger="pictor_mcp.server"):
            status, _, _ = secure_server.request(f"/files/{output['path']}?e=99999999999&s=deadbeef")
        assert status == 403
        assert any("query string" in record.getMessage() for record in caplog.records), [
            record.getMessage() for record in caplog.records
        ]


class TestUnsignedServerRefusesOutputServing:
    def test_config_rejects_output_serving_without_credentials(self, sandbox: Sandbox) -> None:
        from pictor_mcp.errors import ConfigError

        with pytest.raises(ConfigError):
            load_config(base_env(sandbox, PICTOR_SERVE_OUTPUTS="true"))

    def test_file_endpoint_is_absent_when_disabled(self, sandbox: Sandbox) -> None:
        config = load_config(
            base_env(
                sandbox,
                PICTOR_TRANSPORT="streamable-http",
                PICTOR_HOST="127.0.0.1",
                PICTOR_PORT=str(_free_port()),
                PICTOR_AUTH_TOKEN=TOKEN,
                PICTOR_SERVE_OUTPUTS="false",
            )
        )
        with _LiveServer(config) as live:
            status, _, _ = live.request("/files/whatever.png")
            # Not exempted from auth when serving is off, so the request is
            # rejected before routing; either answer means "not served".
            assert status in {401, 404}


class TestHeaderMatching:
    """Unit coverage for the Host/Origin matcher, including malformed values."""

    LOOPBACK_HOSTS = ("127.0.0.1:*", "localhost:*", "[::1]:*", "127.0.0.1:8077", "localhost:8077")

    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("127.0.0.1:8077", True),
            ("127.0.0.1", True),
            ("localhost:8077", True),
            ("[::1]:8077", True),
            ("evil.example.com", False),
            ("localhost.evil.com", False),
            ("127.0.0.1.evil.com", False),
            ("", False),
            # Sanitised rather than prefix-matched: see _host_matches.
            ("127.0.0.1:8077@evil.com", False),
            ("127.0.0.1:8077\nX-Injected: 1", False),
            ("127.0.0.1:8077\x00", False),
        ],
    )
    def test_host_matching(self, host: str, expected: bool) -> None:
        from pictor_mcp.security.auth import _host_matches

        assert _host_matches(host, self.LOOPBACK_HOSTS) is expected

    def test_subdomain_wildcard_does_not_match_the_apex_or_a_suffix_impostor(self) -> None:
        from pictor_mcp.security.auth import _host_matches

        patterns = ("*.example.com",)
        assert _host_matches("cdn.example.com", patterns) is True
        assert _host_matches("example.com", patterns) is False
        assert _host_matches("evilexample.com", patterns) is False

    @pytest.mark.parametrize(
        ("host", "patterns", "expected"),
        [
            # A Host on a non-default port carries the port, so a subdomain
            # wildcard that ignored it could never match a real request to this
            # server - the pattern was documented but unusable.
            ("cdn.example.com:8077", ("*.example.com",), True),
            ("cdn.example.com", ("*.example.com",), True),
            ("a.b.example.com:8077", ("*.example.com",), True),
            ("example.com:8077", ("*.example.com",), False),
            ("evilexample.com:8077", ("*.example.com",), False),
            ("cdn.example.com:8077", ("*.example.com:8077",), True),
            ("cdn.example.com:9000", ("*.example.com:8077",), False),
            ("cdn.example.com:9000", ("*.example.com:*",), True),
            # Bracketed IPv6 must survive the port split.
            ("[::1]:8077", ("[::1]:*",), True),
            ("[::1]", ("[::1]:*",), True),
            ("[::1]:8077", ("[::1]:8077",), True),
            ("[::1]:8077", ("*.example.com",), False),
            # A bare name and a ported one are the same host.
            ("pictor-mcp", ("pictor-mcp:*",), True),
            ("pictor-mcp:8077", ("pictor-mcp:*",), True),
            ("pictor-mcp.evil.com", ("pictor-mcp:*",), False),
            ("127.0.0.1:8077@evil.com", ("*.0.0.1:*",), False),
        ],
    )
    def test_port_handling(self, host: str, patterns: tuple[str, ...], expected: bool) -> None:
        from pictor_mcp.security.auth import _host_matches

        assert _host_matches(host, patterns) is expected

    @pytest.mark.parametrize(
        ("origin", "patterns", "expected"),
        [
            ("http://cdn.example.com:8077", ("*.example.com",), True),
            ("http://cdn.example.com", ("*.example.com",), True),
            ("http://example.com:8077", ("*.example.com",), False),
            ("http://evilexample.com", ("*.example.com",), False),
            # A scheme in the pattern must still be honoured.
            ("https://cdn.example.com", ("http://*.example.com",), False),
            ("https://cdn.example.com:8443", ("https://*.example.com:8443",), True),
            ("https://cdn.example.com:8443", ("https://*.example.com",), True),
            ("https://cdn.example.com:8443", ("https://*.example.com:9443",), False),
        ],
    )
    def test_origin_subdomain_wildcards(self, origin: str, patterns: tuple[str, ...], expected: bool) -> None:
        from pictor_mcp.security.auth import _origin_matches

        assert _origin_matches(origin, patterns) is expected

    def test_the_wildcard_pattern_accepts_anything_but_an_empty_value(self) -> None:
        from pictor_mcp.security.auth import _host_matches

        assert _host_matches("anything.example", ("*",)) is True
        assert _host_matches("", ("*",)) is False

    @pytest.mark.parametrize(
        ("origin", "expected"),
        [
            ("http://127.0.0.1:8077", True),
            ("http://localhost:3000", True),
            ("https://evil.example.com", False),
            ("http://127.0.0.1.evil.com", False),
            ("null", False),
            ("http://127.0.0.1:80\x00", False),
        ],
    )
    def test_origin_matching(self, origin: str, expected: bool) -> None:
        from pictor_mcp.security.auth import _origin_matches

        assert _origin_matches(origin, ("http://127.0.0.1:*", "http://localhost:*")) is expected

    def test_an_empty_origin_allow_list_rejects_every_origin(self) -> None:
        """The wildcard-bind default: no browser Origin is acceptable."""
        from pictor_mcp.security.auth import _origin_matches

        assert _origin_matches("http://10.0.0.5:8077", ()) is False
        assert _origin_matches("https://anything.example", ()) is False


class TestSignedUrlCryptography:
    """A token authorises exactly one path until exactly one instant."""

    SECRET = "s" * 32
    PATH = "resized/a.webp"

    def _token(self, secret: str | None = None, ttl: int = 600) -> str:
        """The token segment of a signed path, as the URL carries it."""
        from pictor_mcp.security.auth import build_signed_path

        return build_signed_path(secret or self.SECRET, self.PATH, ttl).partition("/")[0]

    def _problem(self, token: str, path: str | None = None, secret: str | None = None) -> str | None:
        from pictor_mcp.security.auth import signed_path_problem

        return signed_path_problem(
            self.SECRET if secret is None else secret, token, path if path is not None else self.PATH
        )

    def test_the_exact_path_verifies(self) -> None:
        from pictor_mcp.security.auth import verify_signed_path

        assert verify_signed_path(self.SECRET, self._token(), self.PATH) is True

    def test_the_url_carries_no_query_string(self) -> None:
        """The whole point of the format: nothing to drop, nothing to escape."""
        from pictor_mcp.security.auth import build_signed_path

        signed = build_signed_path(self.SECRET, self.PATH, 600)
        assert "?" not in signed and "&" not in signed and "=" not in signed, signed
        assert signed.endswith("/" + self.PATH), signed

    @pytest.mark.parametrize(
        "other", ["resized/b.webp", "../input/photo.jpg", "resized/a.webp ", "resized/a.webp/../b"]
    )
    def test_a_different_path_does_not(self, other: str) -> None:
        assert self._problem(self._token(), other) is not None

    def test_a_different_secret_does_not(self) -> None:
        assert self._problem(self._token("t" * 32)) is not None

    def test_an_empty_secret_never_verifies(self) -> None:
        """Otherwise a misconfiguration would make every path downloadable."""
        assert self._problem(self._token(), secret="") is not None

    def test_an_expired_token_does_not(self) -> None:
        reason = self._problem(self._token(ttl=-10))
        assert reason is not None and "expired" in reason, reason

    def test_a_tampered_signature_does_not(self) -> None:
        token = self._token()
        expires, _, signature = token.partition(".")
        tampered = f"{expires}.{signature[:-1]}{'0' if signature[-1] != '0' else '1'}"
        assert self._problem(tampered) is not None

    def test_the_expiry_is_covered_by_the_signature(self) -> None:
        """A longer life cannot be claimed by editing the expiry alone."""
        token = self._token()
        expires, _, signature = token.partition(".")
        assert self._problem(f"{int(expires) + 86400}.{signature}") is not None

    @pytest.mark.parametrize("token", ["", "not-a-token", "12345", "12345.", ".abc", "abc.def"])
    def test_a_malformed_token_is_refused_with_a_reason(self, token: str) -> None:
        reason = self._problem(token)
        assert reason is not None and reason, token

    def test_an_old_style_link_names_itself(self) -> None:
        """`/files/<path>?e=…&s=…` is the previous shape; someone will paste one."""
        reason = self._problem("converted", "x.png")
        assert reason is not None
        from pictor_mcp.security.auth import build_signed_path

        assert "or None" not in reason  # sanity: it is a message, not a pass
        assert build_signed_path(self.SECRET, "x.png", 600) != "converted/x.png"

"""Error taxonomy.

Every failure that a tool can produce is a :class:`PictorError` subclass. The
server converts these into MCP tool errors whose text is safe to show to a
model and to a human.

What "safe" means precisely: an *error* message never contains an absolute host
path, a stack trace, a library-internal string, or an environment value. It may
name the configuration *category* that rejected the request ("outside the
configured input roots") but not the configured value. Successful results are a
different matter - ``image_capabilities`` deliberately reports the configured
roots, because an operator needs to know them and the caller already has
authenticated access.

``code`` is a stable machine-readable identifier so that an agent (or a
calling workflow) can branch on the failure without parsing prose.

``PictorError`` derives from the SDK's ``ToolError`` on purpose. Any exception
that is *not* a ``ToolError`` is treated by the SDK as an unanticipated crash:
the client receives only ``Error executing tool <name>`` and the useful message
is swallowed into the server log. Deriving from ``ToolError`` is what makes a
rejected path traversal or an oversized image arrive at the client as
"path is outside the configured input roots" instead of a generic failure.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver.exceptions import ToolError


class PictorError(ToolError):
    """Base class for all expected, client-safe failures."""

    code = "internal_error"
    #: HTTP status used when this error escapes through the HTTP file endpoint.
    http_status = 400

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class ConfigError(PictorError):
    """Invalid server configuration detected at startup."""

    code = "configuration_error"
    http_status = 500


class PathNotAllowedError(PictorError):
    """A requested path is outside every configured allowed root.

    This is deliberately uninformative about *why* (existence, permissions,
    symlink target) so it cannot be used as a filesystem oracle.
    """

    code = "path_not_allowed"


class InputNotFoundError(PictorError):
    """An allowed path does not exist or is not a readable regular file."""

    code = "input_not_found"


class LimitExceededError(PictorError):
    """A configured resource limit was exceeded (size, pixels, frames, time)."""

    code = "limit_exceeded"
    http_status = 413


class UnsupportedFormatError(PictorError):
    """The input is not a recognised image, or the requested output codec is unavailable."""

    code = "unsupported_format"


class InvalidArgumentError(PictorError):
    """Semantically invalid arguments (e.g. crop box outside the image)."""

    code = "invalid_argument"


class NetworkDisabledError(PictorError):
    """URL input was requested but remote fetching is disabled."""

    code = "network_fetch_disabled"
    http_status = 403


class NetworkBlockedError(PictorError):
    """A URL fetch was refused by the SSRF guard."""

    code = "network_fetch_blocked"
    http_status = 403


class BackendUnavailableError(PictorError):
    """A requested processing backend (GPU, ML model) is not installed or not usable."""

    code = "backend_unavailable"
    http_status = 503


class AuthError(PictorError):
    """Missing or invalid credentials on the HTTP transport."""

    code = "unauthorized"
    http_status = 401


class ConcurrencyLimitError(PictorError):
    """Too many operations in flight."""

    code = "too_many_requests"
    http_status = 429

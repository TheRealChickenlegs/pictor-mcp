"""Path confinement tests.

These are the highest-value tests in the suite: an image server that will read
any path an agent names is an arbitrary-file-read primitive. Every case here
corresponds to a concrete escape technique.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pictor_mcp.errors import InputNotFoundError, PathNotAllowedError
from pictor_mcp.security.paths import PathJail, safe_filename, safe_join_uri

from .conftest import Sandbox, base_env


class TestReadConfinement:
    def test_reads_a_file_inside_the_root(self, jail: PathJail) -> None:
        assert jail.read_bytes("photo.jpg", max_bytes=10_000_000)

    def test_reads_a_nested_file(self, jail: PathJail) -> None:
        assert jail.read_bytes("nested/deep.png", max_bytes=10_000_000)

    def test_accepts_an_absolute_path_inside_the_root(self, jail: PathJail, sandbox: Sandbox) -> None:
        assert jail.read_bytes(sandbox.inputs("photo.jpg"), max_bytes=10_000_000)

    @pytest.mark.parametrize(
        "attack",
        [
            "../etc/passwd",
            "../../etc/passwd",
            "nested/../../etc/passwd",
            "nested/../../../../../../etc/shadow",
            "/etc/passwd",
            "/etc/shadow",
            "/proc/self/environ",
            "/proc/self/cmdline",
            "..%2f..%2fetc%2fpasswd",  # not decoded here, so it must 404, not escape
        ],
    )
    def test_rejects_traversal_and_absolute_escapes(self, jail: PathJail, attack: str) -> None:
        with pytest.raises((PathNotAllowedError, InputNotFoundError)):
            jail.read_bytes(attack, max_bytes=10_000_000)

    def test_rejects_null_bytes(self, jail: PathJail) -> None:
        with pytest.raises(PathNotAllowedError):
            jail.read_bytes("photo.jpg\x00.png", max_bytes=10_000_000)

    def test_rejects_control_characters(self, jail: PathJail) -> None:
        with pytest.raises(PathNotAllowedError):
            jail.read_bytes("photo\n.jpg", max_bytes=10_000_000)

    def test_rejects_backslash_paths(self, jail: PathJail) -> None:
        with pytest.raises(PathNotAllowedError):
            jail.read_bytes("..\\..\\etc\\passwd", max_bytes=10_000_000)

    def test_rejects_empty_path(self, jail: PathJail) -> None:
        with pytest.raises(PathNotAllowedError):
            jail.read_bytes("   ", max_bytes=10_000_000)

    def test_rejects_overlong_path(self, jail: PathJail) -> None:
        with pytest.raises(PathNotAllowedError):
            jail.read_bytes("a" * 5000, max_bytes=10_000_000)

    def test_missing_file_reports_not_found(self, jail: PathJail) -> None:
        with pytest.raises(InputNotFoundError):
            jail.read_bytes("nope.jpg", max_bytes=10_000_000)

    def test_rejects_a_directory(self, jail: PathJail) -> None:
        with pytest.raises((PathNotAllowedError, InputNotFoundError, IsADirectoryError)):
            jail.read_bytes("nested", max_bytes=10_000_000)

    def test_enforces_the_size_cap(self, jail: PathJail) -> None:
        from pictor_mcp.errors import LimitExceededError

        with pytest.raises(LimitExceededError):
            jail.read_bytes("photo.jpg", max_bytes=64)

    def test_rejects_a_fifo(self, jail: PathJail, sandbox: Sandbox) -> None:
        """Special files must never be opened: a FIFO would block forever."""
        fifo = sandbox.inputs("pipe")
        os.mkfifo(fifo)
        with pytest.raises(PathNotAllowedError):
            jail.read_bytes("pipe", max_bytes=1024)

    def test_rejects_a_symlink_to_a_symlink_outside(self, jail: PathJail, sandbox: Sandbox) -> None:
        outside = sandbox.root / "outside.txt"
        outside.write_text("secret")
        link = sandbox.inputs("escape.txt")
        link.symlink_to(outside)
        with pytest.raises(PathNotAllowedError):
            jail.read_bytes("escape.txt", max_bytes=1024)

    def test_rejects_a_directory_symlink_escape(self, jail: PathJail, sandbox: Sandbox) -> None:
        outside_dir = sandbox.root / "outside"
        outside_dir.mkdir()
        (outside_dir / "secret.txt").write_text("secret")
        link = sandbox.inputs("linkdir")
        link.symlink_to(outside_dir, target_is_directory=True)
        with pytest.raises(PathNotAllowedError):
            jail.read_bytes("linkdir/secret.txt", max_bytes=1024)

    def test_allows_a_symlink_that_stays_inside(self, jail: PathJail, sandbox: Sandbox) -> None:
        """An internal symlink is legitimate; only escapes are refused."""
        link = sandbox.inputs("alias.png")
        link.symlink_to(sandbox.inputs("photo.png"))
        assert jail.read_bytes("alias.png", max_bytes=10_000_000)

    def test_error_message_does_not_leak_the_filesystem_layout(self, jail: PathJail, sandbox: Sandbox) -> None:
        """A confinement failure must not reveal what exists on the host."""
        with pytest.raises(PathNotAllowedError) as excinfo:
            jail.read_bytes("/etc/passwd", max_bytes=1024)
        message = str(excinfo.value)
        assert "/etc/passwd" not in message
        assert str(sandbox.root) not in message

    def test_unreadable_file_is_reported_without_detail(self, jail: PathJail, sandbox: Sandbox) -> None:
        if os.geteuid() == 0:
            pytest.skip("root can read anything; permission check is meaningless")
        secret = sandbox.inputs("locked.png")
        secret.write_bytes(b"\x89PNG\r\n\x1a\n")
        secret.chmod(0o000)
        try:
            with pytest.raises((PathNotAllowedError, InputNotFoundError)):
                jail.read_bytes("locked.png", max_bytes=1024)
        finally:
            secret.chmod(0o600)


class TestWriteConfinement:
    def test_writes_inside_the_output_root(self, jail: PathJail, sandbox: Sandbox) -> None:
        written = jail.write_bytes("sub/out.bin", b"data")
        assert written.read_bytes() == b"data"
        assert written.is_relative_to(sandbox.output)

    def test_rejects_traversal_out_of_the_output_root(self, jail: PathJail) -> None:
        with pytest.raises(PathNotAllowedError):
            jail.write_bytes("../escaped.bin", b"data")

    def test_rejects_absolute_paths_outside_the_output_root(self, jail: PathJail) -> None:
        with pytest.raises(PathNotAllowedError):
            jail.write_bytes("/tmp/pictor-escape.bin", b"data")

    def test_rejects_writing_over_the_root_itself(self, jail: PathJail) -> None:
        with pytest.raises(PathNotAllowedError):
            jail.write_bytes(".", b"data")

    def test_refuses_to_follow_a_symlinked_parent(self, jail: PathJail, sandbox: Sandbox) -> None:
        """A symlink planted in the output dir must not redirect the write."""
        outside_dir = sandbox.root / "elsewhere"
        outside_dir.mkdir()
        link = sandbox.outputs("sneaky")
        link.symlink_to(outside_dir, target_is_directory=True)
        with pytest.raises(PathNotAllowedError):
            jail.write_bytes("sneaky/planted.bin", b"data")
        assert not (outside_dir / "planted.bin").exists()

    def test_refuses_to_overwrite_when_asked_not_to(self, jail: PathJail) -> None:
        jail.write_bytes("keep.bin", b"first")
        with pytest.raises(PathNotAllowedError):
            jail.write_bytes("keep.bin", b"second", overwrite=False)
        assert jail.read_output_bytes("keep.bin") == b"first"

    def test_write_is_atomic_leaving_no_temp_files(self, jail: PathJail, sandbox: Sandbox) -> None:
        for index in range(5):
            jail.write_bytes(f"many/file-{index}.bin", b"x" * 128)
        leftovers = [p.name for p in (sandbox.output / "many").iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []

    def test_read_output_bytes_is_confined(self, jail: PathJail) -> None:
        jail.write_bytes("ok.bin", b"fine")
        assert jail.read_output_bytes("ok.bin") == b"fine"
        with pytest.raises(PathNotAllowedError):
            jail.read_output_bytes("../input/photo.jpg")

    def test_relative_output_is_posix_style(self, jail: PathJail) -> None:
        written = jail.write_bytes("web/thing.webp", b"x")
        assert jail.relative_output(written) == "web/thing.webp"


class TestSafeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("photo", "photo.png"),
            ("a/b/c", "a_b_c.png"),
            ("..", "image.png"),
            ("", "image.png"),
            (".hidden", "hidden.png"),
            ("con", "_con.png"),
            ("name\x00evil", "name_evil.png"),
        ],
    )
    def test_sanitises_untrusted_names(self, raw: str, expected: str) -> None:
        assert safe_filename(raw, ".png") == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "../../etc/passwd",
            "/etc/passwd",
            "....//....//etc/shadow",
            "a/../../b",
            "..\\..\\windows\\system32",
            "\x00/etc/passwd",
            "  ..  ",
            "con",
        ],
    )
    def test_never_produces_a_traversing_name(self, raw: str) -> None:
        """The exact spelling does not matter; these properties do."""
        name = safe_filename(raw, ".png")
        assert "/" not in name
        assert "\\" not in name
        assert "\x00" not in name
        assert not name.startswith(".")
        assert ".." not in Path(name).parts
        # Resolving it inside a directory must stay inside that directory.
        resolved = (Path("/base") / name).resolve()
        assert resolved.parent == Path("/base").resolve()

    def test_rejects_a_separator_in_the_extension(self) -> None:
        with pytest.raises(PathNotAllowedError):
            safe_filename("name", ".p/ng")

    def test_truncates_long_names(self) -> None:
        assert len(safe_filename("x" * 500, ".png")) <= 100


class TestSafeJoinUri:
    def test_joins_inside_the_root(self, tmp_path: Path) -> None:
        assert safe_join_uri(tmp_path, "a/b.png") == (tmp_path / "a" / "b.png").resolve()

    @pytest.mark.parametrize("relative", ["../x", "a/../../x", "/etc/passwd", "a\x00b", "a\\b"])
    def test_refuses_escapes(self, tmp_path: Path, relative: str) -> None:
        with pytest.raises(PathNotAllowedError):
            safe_join_uri(tmp_path, relative)


class TestConfigGuards:
    def test_output_root_may_not_be_an_input_root(self, sandbox: Sandbox) -> None:
        from pictor_mcp.config import load_config
        from pictor_mcp.errors import ConfigError

        with pytest.raises(ConfigError):
            load_config(
                {
                    "PICTOR_INPUT_ROOTS": str(sandbox.input),
                    "PICTOR_OUTPUT_ROOT": str(sandbox.input),
                }
            )

    def test_relative_roots_are_refused(self) -> None:
        from pictor_mcp.config import load_config
        from pictor_mcp.errors import ConfigError

        with pytest.raises(ConfigError):
            load_config({"PICTOR_INPUT_ROOTS": "relative/path"})

    def test_serving_outputs_without_any_credential_is_refused(self, sandbox: Sandbox) -> None:
        """Fail closed rather than exposing every generated file."""
        from pictor_mcp.config import load_config
        from pictor_mcp.errors import ConfigError

        with pytest.raises(ConfigError):
            load_config(base_env(sandbox, PICTOR_SERVE_OUTPUTS="true"))

    def test_serving_outputs_with_a_token_is_allowed(self, sandbox: Sandbox) -> None:
        from pictor_mcp.config import load_config

        config = load_config(
            base_env(
                sandbox,
                PICTOR_SERVE_OUTPUTS="true",
                PICTOR_AUTH_TOKEN="a-sufficiently-long-token-value",
            )
        )
        assert config.http.serve_outputs is True

    def test_short_auth_token_is_refused(self, sandbox: Sandbox) -> None:
        from pictor_mcp.config import load_config
        from pictor_mcp.errors import ConfigError

        with pytest.raises(ConfigError):
            load_config(base_env(sandbox, PICTOR_AUTH_TOKEN="short"))

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("PICTOR_MAX_PIXELS", "0"),
            ("PICTOR_MAX_CONCURRENCY", "0"),
            ("PICTOR_PORT", "99999"),
            ("PICTOR_TRANSPORT", "carrier-pigeon"),
            ("PICTOR_ALLOW_NET_FETCH", "maybe"),
            ("PICTOR_GPU", "quantum"),
            ("PICTOR_STREAMABLE_HTTP_PATH", "no-leading-slash"),
        ],
    )
    def test_invalid_values_fail_loudly(self, sandbox: Sandbox, key: str, value: str) -> None:
        """A silently ignored security setting is worse than a server that will not boot."""
        from pictor_mcp.config import load_config
        from pictor_mcp.errors import ConfigError

        with pytest.raises(ConfigError):
            load_config(base_env(sandbox, **{key: value}))

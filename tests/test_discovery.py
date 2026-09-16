"""Tests for image_list_inputs, the tool that answers "what can you read?".

The tool exists because a model told about an image by name - an attachment in a
chat UI - cannot otherwise find it, and guessing produces `input_not_found`,
which reads as a broken server. The property that matters most is therefore not
the formatting: it is that every `path` the tool reports can be handed straight
back to another tool and resolve to the file it described. A listing that lies
about that is worse than no listing.
"""

from __future__ import annotations

import os
import time

import anyio
import pytest
from mcp.server.mcpserver import MCPServer

from pictor_mcp.config import load_config
from pictor_mcp.server import build_context
from pictor_mcp.tools import discovery
from pictor_mcp.tools.context import ToolContext

from .conftest import Sandbox, base_env


def _context(sandbox: Sandbox, **overrides: str) -> ToolContext:
    """The real context builder, so the test cannot wire it differently."""
    return build_context(load_config(base_env(sandbox, **overrides)))


def _listing(ctx: ToolContext, **kwargs: object) -> dict:
    """Call the tool through a real server and return its structured result."""
    server = MCPServer(name="test")
    discovery.register(server, ctx)
    result = anyio.run(server.call_tool, "image_list_inputs", dict(kwargs))
    assert result.structured_content is not None, result
    return result.structured_content


@pytest.fixture
def listing(sandbox: Sandbox):
    ctx = _context(sandbox)
    return lambda **kwargs: _listing(ctx, **kwargs)


class TestListing:
    def test_newest_first(self, sandbox: Sandbox, listing) -> None:
        """The image someone just attached is the one they mean."""
        older = sandbox.inputs("photo.jpg")
        newer = sandbox.inputs("logo.png")
        os.utime(older, (time.time() - 600, time.time() - 600))
        os.utime(newer, (time.time(), time.time()))

        result = listing()
        paths = [entry["path"] for entry in result["files"]]
        assert paths[0] == "logo.png", paths
        assert "photo.jpg" in paths

    def test_every_reported_path_resolves_back_to_its_file(self, sandbox: Sandbox) -> None:
        """The whole point: the answer must be usable by the next tool."""
        ctx = _context(sandbox)
        result = _listing(ctx)
        assert result["files"], "nothing listed"
        for entry in result["files"]:
            assert ctx.jail.resolve_read(entry["path"]).name == entry["name"], entry

    def test_reports_size_and_time(self, sandbox: Sandbox, listing) -> None:
        result = listing(pattern="photo.jpg")
        entry = result["files"][0]
        assert entry["byteSize"] == sandbox.inputs("photo.jpg").stat().st_size
        assert entry["modified"].endswith("Z") and "T" in entry["modified"], entry["modified"]

    def test_lists_nested_files_by_relative_path(self, sandbox: Sandbox, listing) -> None:
        paths = [entry["path"] for entry in listing()["files"]]
        assert "nested/deep.png" in paths, paths

    def test_a_glob_filter_is_case_insensitive(self, sandbox: Sandbox, listing) -> None:
        upper = listing(pattern="*.PNG")["files"]
        lower = listing(pattern="*.png")["files"]
        assert [entry["path"] for entry in upper] == [entry["path"] for entry in lower]
        assert upper, "the filter matched nothing"

    def test_the_limit_bounds_the_result_and_says_so(self, sandbox: Sandbox, listing) -> None:
        result = listing(limit=2)
        assert len(result["files"]) == 2
        assert result["truncated"] is True

    def test_the_scan_is_bounded(self, sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, listing) -> None:
        """A root with a million files must cost a bounded amount of work."""
        monkeypatch.setattr(discovery, "_MAX_SCAN", 2)
        result = listing(limit=200)
        assert result["scanned"] <= 2
        assert result["truncated"] is True

    def test_hidden_entries_are_not_offered(self, sandbox: Sandbox, listing) -> None:
        """Dotfiles under a root are someone's tooling, not images for a model."""
        (sandbox.input / ".hidden.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        hidden_dir = sandbox.input / ".cache"
        hidden_dir.mkdir()
        (hidden_dir / "cached.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        paths = [entry["path"] for entry in listing()["files"]]
        assert not [path for path in paths if ".hidden" in path or ".cache" in path], paths

    def test_a_symlink_is_not_listed(self, sandbox: Sandbox, listing) -> None:
        """Reads open the final component with O_NOFOLLOW, so a listed symlink
        would be a path that cannot be used."""
        (sandbox.input / "link.png").symlink_to(sandbox.inputs("photo.jpg"))
        paths = [entry["path"] for entry in listing()["files"]]
        assert "link.png" not in paths, paths

    def test_a_symlinked_directory_is_not_descended(self, sandbox: Sandbox, listing) -> None:
        outside = sandbox.root / "outside"
        outside.mkdir()
        (outside / "secret.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (sandbox.input / "escape").symlink_to(outside, target_is_directory=True)
        paths = [entry["path"] for entry in listing()["files"]]
        assert not [path for path in paths if "secret" in path], paths

    def test_a_fifo_is_not_listed(self, sandbox: Sandbox, listing) -> None:
        """Opening one blocks until a peer appears; it is not an image either."""
        os.mkfifo(sandbox.input / "pipe.png")
        paths = [entry["path"] for entry in listing()["files"]]
        assert "pipe.png" not in paths, paths

    def test_the_output_root_is_opt_in(self, sandbox: Sandbox, listing) -> None:
        sandbox.outputs("resized").mkdir()
        sandbox.outputs("resized/out.webp").write_bytes(b"RIFF")
        assert "resized/out.webp" not in [entry["path"] for entry in listing()["files"]]
        included = listing(include_outputs=True)
        assert "resized/out.webp" in [entry["path"] for entry in included["files"]]

    def test_roots_are_reported(self, sandbox: Sandbox, listing) -> None:
        result = listing()
        assert str(sandbox.input) in result["roots"]

    def test_an_empty_root_is_not_an_error(self, sandbox: Sandbox, listing) -> None:
        result = listing(pattern="nothing-matches-this")
        assert result["files"] == []
        assert result["truncated"] is False


class TestShadowedNames:
    """Two roots can hold the same relative name; the listing must not lie."""

    def test_a_shadowed_file_is_reported_by_its_absolute_path(self, sandbox: Sandbox) -> None:
        second = sandbox.root / "second"
        second.mkdir()
        (second / "photo.jpg").write_bytes(sandbox.inputs("photo.jpg").read_bytes())
        # The first root wins for a relative path, so the second root's file is
        # only reachable by absolute path - and that is what must be reported.
        ctx = _context(sandbox, PICTOR_INPUT_ROOTS=f"{sandbox.input},{second}")
        result = _listing(ctx)

        shadowed = [entry for entry in result["files"] if entry["root"] == str(second)]
        assert shadowed, result["files"]
        assert shadowed[0]["path"].startswith("/"), shadowed[0]
        for entry in result["files"]:
            assert ctx.jail.resolve_read(entry["path"]).name == entry["name"], entry


class TestRegistration:
    def test_the_tool_is_part_of_the_flat_tool_set(self) -> None:
        import inspect

        from pictor_mcp.tools import register_all

        assert "image_list_inputs" in inspect.getsource(register_all)

    def test_the_listing_decodes_nothing(self, sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
        """A truncated file is not an error here: metadata needs no decode."""
        ctx = _context(sandbox)
        result = _listing(ctx)
        names = {entry["name"] for entry in result["files"]}
        assert "broken.jpg" in names and "notimage.txt" in names, names


def _tool_names(sandbox: Sandbox) -> set[str]:
    """The names a client actually sees."""
    ctx = _context(sandbox)
    server = MCPServer(name="test")
    from pictor_mcp.tools import register_all

    register_all(server, ctx)
    return {tool.name for tool in anyio.run(server.list_tools)}


class TestToolSurface:
    def test_the_new_tool_is_visible_to_clients(self, sandbox: Sandbox) -> None:
        assert "image_list_inputs" in _tool_names(sandbox)

    def test_the_registry_and_the_server_agree(self, sandbox: Sandbox) -> None:
        """`register_all` returns a hand-written name list; drift here shows up as
        a client that cannot call a tool the server believes it has."""
        ctx = _context(sandbox)
        server = MCPServer(name="test")
        from pictor_mcp.tools import register_all

        returned = set(register_all(server, ctx))
        assert returned == _tool_names(sandbox), returned ^ _tool_names(sandbox)

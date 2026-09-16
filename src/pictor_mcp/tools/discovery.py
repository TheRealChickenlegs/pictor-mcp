"""Discovering what the server can read.

Every other tool takes a `path` the caller has to know already. That is fine for
a client that has been told a filename, and useless for the case this module
exists for: an image attached in a chat UI, which the model knows about by name
or by file id but which lives under a storage path it has never seen. Without a
way to look, the model guesses - and a guess produces `input_not_found`, which
looks like a broken server rather than a missing answer.

So: one tool that lists what is readable, newest first, metadata only. It decodes
nothing, so it stays cheap enough to call before every operation, and it cannot
be used to make the server do work. Everything it lists is confined by the same
path jail the read tools use, and every `path` it returns is verified to resolve
back to the file it describes.
"""

from __future__ import annotations

import os
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..config import Config
from ..errors import PictorError
from ..models import InputFile, InputListing
from ..outputs import human_bytes
from ..security.paths import PathJail
from .context import ToolContext, tool_guard

#: Directory levels descended below a root. Deep enough for date- or
#: collection-organised folders, shallow enough that a root pointing at
#: something enormous cannot turn a listing into a filesystem walk.
_MAX_DEPTH = 4

#: Directory entries examined per call. The cap is on the *scan*, not the
#: result, so a root with a million files costs a bounded amount of work and
#: says it was cut short rather than stalling the request.
_MAX_SCAN = 5000

#: Hard ceiling on returned entries, whatever the caller asks for.
_MAX_LIMIT = 200


def _candidate(entry: os.DirEntry[str]) -> os.stat_result | None:
    """Stat an entry only if it is a regular file this server could later open.

    Symlinks are skipped rather than resolved: reads open the final component
    with ``O_NOFOLLOW``, so a listed symlink would be a path that cannot be
    used. Hidden entries are skipped too - dotfiles under an input root are
    someone's tooling, not an image to offer a model.
    """
    if entry.name.startswith("."):
        return None
    try:
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            return None
        return entry.stat(follow_symlinks=False)
    except OSError:
        # A file that vanished mid-scan, or one we may not stat. Either way it
        # is not something to promise the caller.
        return None


def _walk(root: Path, *, pattern: str | None, budget: int) -> tuple[list[tuple[Path, os.stat_result]], int, bool]:
    """Collect regular files under ``root``, newest first, within ``budget``."""
    found: list[tuple[Path, os.stat_result]] = []
    scanned = 0
    truncated = False

    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            if scanned >= budget:
                return found, scanned, True
            scanned += 1
            if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                if depth < _MAX_DEPTH:
                    stack.append((Path(entry.path), depth + 1))
                else:
                    truncated = True
                continue
            stat_result = _candidate(entry)
            if stat_result is None:
                continue
            if pattern and not fnmatch(entry.name.lower(), pattern):
                continue
            found.append((Path(entry.path), stat_result))

    found.sort(key=lambda item: item[1].st_mtime, reverse=True)
    return found, scanned, truncated


def _describe(jail: PathJail, root: Path, path: Path) -> InputFile | None:
    """Build one listing entry, with a path that provably resolves back to it.

    Relative paths are tried against every readable root in order, so the same
    name in two roots is shadowed: the second file would be listed under a path
    that reads the first. Where the relative form does not round-trip, the
    absolute path is reported instead - the jail accepts an absolute path inside
    a root, so the caller still gets something usable.
    """
    try:
        stat_result = path.stat()
    except OSError:
        return None

    relative = path.relative_to(root)
    reported = str(relative)
    try:
        if jail.resolve_read(reported) != path:
            reported = str(path)
    except PictorError:
        # The jail refused the relative form - it would resolve to a different
        # file, or to nothing. The absolute path is still inside a root, so the
        # caller gets something it can use.
        reported = str(path)

    return InputFile(
        path=reported,
        name=path.name,
        byte_size=stat_result.st_size,
        modified=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat_result.st_mtime)),
        root=str(root),
    )


def _resolve_roots(roots: tuple[Path, ...], outputs: Path, include_outputs: bool) -> list[Path]:
    """Real paths, so a root symlinked to itself does not make comparisons lie."""
    candidates = [*roots, outputs] if include_outputs else list(roots)
    resolved: list[Path] = []
    for candidate in candidates:
        real = Path(os.path.realpath(candidate))
        if real not in resolved:
            resolved.append(real)
    return resolved


def register(server: MCPServer, ctx: ToolContext) -> None:
    """Register the discovery tools on ``server``."""

    @server.tool(
        name="image_list_inputs",
        title="List readable image files",
        description=(
            "List the files this server can read, newest first, with the `path` to pass to the "
            "other tools. Use it to find an image you have been told about by name - an "
            "attachment in a chat UI, for example - instead of guessing at a path or a filename. "
            "Returns metadata only; nothing is decoded, so it is cheap to call first."
        ),
    )
    @tool_guard
    async def image_list_inputs(
        pattern: Annotated[
            str | None,
            Field(
                description=(
                    "Optional glob matched against the filename, case-insensitively, e.g. '*.png' or 'chicken*'."
                )
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(ge=1, le=_MAX_LIMIT, description="Maximum entries to return, newest first."),
        ] = 25,
        include_outputs: Annotated[
            bool,
            Field(description="Also list files the server produced, not just the input roots."),
        ] = False,
    ) -> InputListing:
        import anyio

        from ..outputs import build_result

        config: Config = ctx.config
        roots = _resolve_roots(config.input_roots, config.output_root, include_outputs)
        glob = pattern.lower() if pattern else None

        def work() -> tuple[list[InputFile], int, bool]:
            remaining = _MAX_SCAN
            entries: list[tuple[Path, os.stat_result, Path]] = []
            scanned = 0
            truncated = False
            for root in roots:
                if remaining <= 0:
                    truncated = True
                    break
                found, used, cut = _walk(root, pattern=glob, budget=remaining)
                remaining -= used
                scanned += used
                truncated = truncated or cut
                entries.extend((path, stat_result, root) for path, stat_result in found)

            entries.sort(key=lambda item: item[1].st_mtime, reverse=True)
            if len(entries) > limit:
                truncated = True
            described = [entry for path, _stat, root in entries[:limit] if (entry := _describe(ctx.jail, root, path))]
            return described, scanned, truncated

        files, scanned, truncated = await anyio.to_thread.run_sync(work)

        listing = InputListing(files=files, roots=[str(root) for root in roots], scanned=scanned, truncated=truncated)
        if files:
            header = f"{len(files)} readable file(s), newest first ({scanned} entries scanned):"
        else:
            header = f"No readable files found under {', '.join(str(root) for root in roots)}."
            if pattern:
                header += f" Filter in use: {pattern}."
        lines = [header]
        for entry in files:
            lines.append(f"  {entry.path}  ({human_bytes(entry.byte_size)}, modified {entry.modified})")
        if truncated:
            lines.append("The listing was cut short; narrow it with `pattern` or raise `limit`.")
        if files:
            lines.append("Pass one of those paths as `path` to image_info, image_convert or any other tool.")
        return build_result(listing, "\n".join(lines))

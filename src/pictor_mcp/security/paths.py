"""Filesystem confinement.

An image server is a file-read/file-write primitive wearing a costume. If it
will read any path an agent names, it is an arbitrary-file-read gadget; if it
will write any path, it is an arbitrary-file-write gadget. Both are far more
dangerous than "resize a PNG", so confinement is enforced here rather than left
to tool authors.

Threats handled
---------------
* **Traversal** - ``../../etc/passwd`` and friends. Every candidate is resolved
  and then checked for containment against a configured root.
* **Symlink escape** - a link inside an allowed root pointing outside it.
  Resolution happens *before* the containment check, and the final component is
  opened with ``O_NOFOLLOW``.
* **TOCTOU swaps** - swapping a checked file for a symlink between the check and
  the read. Reads are performed on an already-open descriptor whose real path is
  re-verified through ``/proc/self/fd``.
* **Special files** - ``/dev/zero``, FIFOs and devices, which would hang or
  exhaust memory. Only regular files are opened.
* **Filesystem oracles** - callers learn "not allowed", never whether a
  forbidden path exists, so the server cannot be used to probe the host.
"""

from __future__ import annotations

import errno
import os
import re
import stat
import unicodedata
from pathlib import Path, PurePosixPath

from ..errors import InputNotFoundError, PathNotAllowedError

#: Paths longer than this are rejected outright (Linux PATH_MAX is 4096).
_MAX_PATH_BYTES = 4096

#: Characters that are illegal or dangerous in a generated filename.
_UNSAFE_FILENAME = re.compile(r"[\x00-\x1f\x7f/\\:*?\"<>|]")

#: Reserved device names on Windows; harmless on Linux but cheap to avoid.
_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
)


def _reject_nul_and_control(value: str) -> None:
    if "\x00" in value:
        raise PathNotAllowedError("path contains a NUL byte")
    if any(ord(ch) < 0x20 for ch in value):
        raise PathNotAllowedError("path contains control characters")
    if len(value.encode("utf-8", "surrogatepass")) > _MAX_PATH_BYTES:
        raise PathNotAllowedError("path is too long")


def _is_relative_to(child: Path, parent: Path) -> bool:
    """``Path.is_relative_to`` with a fallback for exotic inputs."""
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


class PathJail:
    """Confines all reads to ``input_roots`` and all writes to ``output_root``."""

    __slots__ = ("_input_roots", "_output_root")

    def __init__(self, input_roots: tuple[Path, ...], output_root: Path) -> None:
        if not input_roots:
            raise ValueError("at least one input root is required")
        self._input_roots = tuple(Path(r) for r in input_roots)
        self._output_root = Path(output_root)

    # ------------------------------------------------------------- accessors
    @property
    def input_roots(self) -> tuple[Path, ...]:
        return self._input_roots

    @property
    def output_root(self) -> Path:
        return self._output_root

    def ensure_output_root(self) -> None:
        """Create the output root with restrictive permissions."""
        self._output_root.mkdir(parents=True, exist_ok=True, mode=0o750)

    # --------------------------------------------------------------- resolve
    def resolve_read(self, raw: str | os.PathLike[str]) -> Path:
        """Resolve ``raw`` to an existing regular file inside a readable root.

        Relative paths are tried against every readable root in order, so a
        caller can say ``"photo.jpg"`` for an input or ``"resized/out.webp"``
        for a file the server produced earlier.
        """
        text = self._validate_text(raw)
        path = Path(text)
        candidates: list[Path] = []
        if path.is_absolute():
            candidates.append(Path(os.path.normpath(path)))
        else:
            for base in self._readable_roots():
                candidates.append(self._normalise(text, base=base))

        roots = [self._resolve_existing(root) for root in self._readable_roots()]
        saw_inside = False
        saw_outside = False
        for candidate in candidates:
            resolved = self._resolve_existing(candidate)
            if not any(_is_relative_to(resolved, root) for root in roots):
                # A candidate resolving outside every readable root is an escape
                # attempt (traversal, or a symlink leading out), not a miss.
                # This is how a symlinked path inside an allowed root is caught.
                saw_outside = True
                continue
            saw_inside = True
            if resolved.exists():
                return resolved

        if saw_outside:
            # Denial wins over "not found": a path that tried to leave the
            # sandbox must be reported as refused, and the later fallback
            # candidates (the other readable roots) must not mask that.
            raise PathNotAllowedError("path is outside the configured input roots")
        if saw_inside:
            raise InputNotFoundError("input file was not found")
        raise PathNotAllowedError("path is outside the configured input roots")

    def _readable_roots(self) -> tuple[Path, ...]:
        """Every directory a read may come from.

        The output root is included deliberately. Reading back a file the server
        itself just produced is part of the normal workflow - resize, then
        compare the result, or run a second pass over it - and the output root is
        server-owned, so this widens convenience without widening the trust
        boundary. Only *writes* stay restricted to the output root alone.
        """
        return (*self._input_roots, self._output_root)

    def resolve_write(self, raw: str | os.PathLike[str]) -> Path:
        """Resolve ``raw`` to a writable path inside the output root.

        The file itself need not exist. Every existing ancestor must already be
        inside the output root, which stops a symlinked parent from redirecting
        the write elsewhere.
        """
        candidate = self._normalise(raw, base=self._output_root)
        root = self._resolve_existing(self._output_root)

        parent = candidate.parent
        # Walk from the root down so a symlinked intermediate component is
        # caught even when the final file does not exist yet.
        existing_parent = self._resolve_existing(parent)
        if not _is_relative_to(existing_parent, root):
            raise PathNotAllowedError("path is outside the configured output root")

        final = existing_parent / candidate.name
        if not _is_relative_to(final, root):
            raise PathNotAllowedError("path is outside the configured output root")
        if final == root:
            raise PathNotAllowedError("path must name a file, not the output root")
        return final

    def make_output_dir(self, raw: str | os.PathLike[str]) -> Path:
        """Resolve *and create* a subdirectory of the output root."""
        directory = self.resolve_write(raw)
        # Re-check after creation: mkdir follows a symlinked parent otherwise.
        directory.mkdir(parents=True, exist_ok=True, mode=0o750)
        rechecked = self._resolve_existing(directory)
        if not _is_relative_to(rechecked, self._resolve_existing(self._output_root)):
            raise PathNotAllowedError("path is outside the configured output root")
        return rechecked

    # ------------------------------------------------------------------ read
    def read_bytes(self, raw: str | os.PathLike[str], *, max_bytes: int) -> bytes:
        """Read a confined file with no TOCTOU window and a hard size cap.

        The file is opened with ``O_NOFOLLOW`` and its identity is re-verified
        through the open descriptor, so a path swapped for a symlink after
        validation cannot redirect the read.
        """
        resolved = self.resolve_read(raw)

        # O_NONBLOCK matters for correctness, not throughput: opening a FIFO or
        # a device O_RDONLY blocks until a peer appears, so without it a FIFO
        # planted in an input root would hang the worker thread forever before
        # the S_ISREG check could ever run. It has no effect on regular files.
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(resolved, flags)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EMLINK}:
                raise PathNotAllowedError("path is a symbolic link") from exc
            if exc.errno in {errno.ENOENT, errno.ENOTDIR}:
                raise InputNotFoundError("input file was not found") from exc
            if exc.errno in {errno.EACCES, errno.EPERM}:
                raise PathNotAllowedError("input file is not readable") from exc
            raise InputNotFoundError("input file could not be opened") from exc

        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise PathNotAllowedError("only regular files may be read")
            if st.st_size > max_bytes:
                from ..errors import LimitExceededError

                raise LimitExceededError(
                    f"file is larger than the {max_bytes} byte limit",
                    size_bytes=st.st_size,
                    limit_bytes=max_bytes,
                )
            self._verify_fd_path(fd)
            data = self._read_bounded(fd, max_bytes)
        finally:
            os.close(fd)

        if not data:
            raise InputNotFoundError("input file is empty")
        return data

    def _read_bounded(self, fd: int, max_bytes: int) -> bytes:
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            from ..errors import LimitExceededError

            raise LimitExceededError(
                f"file grew beyond the {max_bytes} byte limit while reading",
                limit_bytes=max_bytes,
            )
        return data

    def _verify_fd_path(self, fd: int) -> None:
        """Confirm the *opened* file still lives inside an allowed root.

        ``/proc/self/fd`` is the only portable-ish way to ask "what did this
        descriptor actually open?". Where it is unavailable (non-Linux, or a
        hardened procfs) we fall back to the pre-open resolution, which is
        already protected by ``O_NOFOLLOW`` on the final component.
        """
        try:
            real = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            return
        real = re.sub(r" \(deleted\)$", "", real)
        if not real.startswith("/"):
            return  # anonymous fd (memfd, pipe); not a path we handed out
        real_path = Path(real)
        for root in self._readable_roots():
            if _is_relative_to(real_path, self._resolve_existing(root)):
                return
        raise PathNotAllowedError("opened file is outside the configured input roots")

    # ----------------------------------------------------------------- write
    def write_bytes(self, raw: str | os.PathLike[str], data: bytes, *, overwrite: bool = True) -> Path:
        """Write ``data`` to a confined path, atomically.

        The write goes to a temporary file in the destination directory which is
        then renamed into place, so a reader never observes a partial image and
        a failure leaves no half-written file behind.
        """
        target = self.resolve_write(raw)
        if target.exists() and not overwrite:
            raise PathNotAllowedError("refusing to overwrite an existing file")

        directory = target.parent
        directory.mkdir(parents=True, exist_ok=True, mode=0o750)
        # mkdir can traverse a symlinked ancestor; re-validate before writing.
        if not _is_relative_to(self._resolve_existing(directory), self._resolve_existing(self._output_root)):
            raise PathNotAllowedError("path is outside the configured output root")

        tmp_name = f".{target.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
        tmp_path = directory / tmp_name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(tmp_path, flags, 0o640)
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            tmp_path.unlink(missing_ok=True)
            raise
        else:
            os.close(fd)

        try:
            os.replace(tmp_path, target)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        return target

    def relative_output(self, path: Path) -> str:
        """Path relative to the output root, POSIX-style (portable in results)."""
        try:
            return path.resolve().relative_to(self._resolve_existing(self._output_root)).as_posix()
        except ValueError:
            return path.name

    def read_output_bytes(self, relative: str, *, max_bytes: int | None = None) -> bytes:
        """Read back a file previously written under the output root.

        Used when a result must be re-read (for example to inline the first
        image of a batch). The path is resolved inside the output root and the
        same ``O_NOFOLLOW`` + descriptor verification applies, so a symlink
        planted in the output directory cannot turn this into an arbitrary read.
        """
        target = self.resolve_write(relative)
        limit = max_bytes if max_bytes is not None else 256 * 1024 * 1024
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(target, flags)
        except OSError as exc:
            if exc.errno in {errno.ENOENT, errno.ENOTDIR}:
                raise InputNotFoundError("output file was not found") from exc
            raise PathNotAllowedError("output file is not readable") from exc
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise PathNotAllowedError("only regular files may be read")
            if st.st_size > limit:
                from ..errors import LimitExceededError

                raise LimitExceededError("output file is larger than the read limit")
            # Parity with read_bytes: re-verify what the descriptor actually
            # opened, so a symlinked intermediate directory swapped in after
            # validation cannot redirect the read.
            self._verify_fd_path(fd)
            return self._read_bounded(fd, limit)
        finally:
            os.close(fd)

    # ---------------------------------------------------------------- helpers
    def contains(self, path: Path) -> bool:
        """True when ``path`` is inside a readable root."""
        resolved = self._resolve_existing(path)
        return any(_is_relative_to(resolved, self._resolve_existing(root)) for root in self._readable_roots())

    def _validate_text(self, raw: str | os.PathLike[str]) -> str:
        """Normalise and screen a caller-supplied path string."""
        if isinstance(raw, os.PathLike):
            raw = os.fspath(raw)
        if not isinstance(raw, str):
            raise PathNotAllowedError("path must be a string")
        text = raw.strip()
        if not text:
            raise PathNotAllowedError("path must not be empty")
        _reject_nul_and_control(text)
        text = unicodedata.normalize("NFC", text)
        if "\\" in text:
            raise PathNotAllowedError("backslash path separators are not accepted")
        return text

    def _normalise(self, raw: str | os.PathLike[str], *, base: Path) -> Path:
        text = self._validate_text(raw)
        path = Path(text)
        if not path.is_absolute():
            path = base / path
        return Path(os.path.normpath(path))

    @staticmethod
    def _resolve_existing(path: Path) -> Path:
        """``resolve(strict=False)`` without raising on permission errors."""
        try:
            return path.resolve()
        except OSError:
            return Path(os.path.abspath(path))


def safe_filename(stem: str, suffix: str, *, max_length: int = 96) -> str:
    """Build a filesystem-safe filename from untrusted text.

    Used when a caller supplies a name for a derived artefact. Strips path
    separators, control characters, leading dots and reserved stems, and never
    returns an empty name.
    """
    cleaned = _UNSAFE_FILENAME.sub("_", unicodedata.normalize("NFKC", stem)).strip(" .")
    cleaned = re.sub(r"_{2,}", "_", cleaned)
    if cleaned.lower() in _RESERVED_STEMS:
        cleaned = f"_{cleaned}"
    cleaned = cleaned[:max_length].strip(" .") or "image"
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    if _UNSAFE_FILENAME.search(suffix) or "/" in suffix:
        raise PathNotAllowedError("invalid output extension")
    return f"{cleaned}{suffix}"


def safe_join_uri(root: Path, relative: str) -> Path:
    """Join a URI-derived relative path onto ``root``, refusing escapes.

    Mirrors the MCP SDK's ``safe_join`` semantics but raises our own error type
    so resource handlers return a consistent code.
    """
    _reject_nul_and_control(relative)
    if "\\" in relative:
        raise PathNotAllowedError("backslash path separators are not accepted")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise PathNotAllowedError("resource path escapes its root")
    candidate = (root / pure).resolve()
    resolved_root = root.resolve()
    if not _is_relative_to(candidate, resolved_root):
        raise PathNotAllowedError("resource path escapes its root")
    return candidate


__all__ = ["PathJail", "safe_filename", "safe_join_uri"]

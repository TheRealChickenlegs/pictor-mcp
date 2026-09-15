"""Font discovery for text watermarks.

Watermark text is caller-supplied and so is the font name, which makes "load
this font path" an obvious file-read primitive. Instead, fonts are found by
*name* inside a fixed set of directories scanned once at startup, exactly like
a CSS ``font-family`` lookup. A caller can choose ``DejaVuSans-Bold``; it
cannot choose ``/etc/shadow``.

When no font is installed at all, Pillow's built-in scalable default is used so
the feature still works in a minimal image.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import ImageFont

logger = logging.getLogger(__name__)

_FONT_SUFFIXES = frozenset({".ttf", ".otf", ".ttc", ".otc"})

#: Preference order when the caller does not name a font. DejaVu is the most
#: commonly packaged; Liberation and Noto are the usual substitutes.
_PREFERRED = (
    "dejavusans",
    "liberationsans",
    "notosans",
    "freesans",
    "arial",
    "helvetica",
)


@dataclass(slots=True)
class FontIndex:
    """Name -> path index over the configured font directories."""

    directories: tuple[Path, ...]
    _index: dict[str, Path] = field(default_factory=dict)
    _scanned: bool = False

    def scan(self) -> None:
        if self._scanned:
            return
        self._scanned = True
        for directory in self.directories:
            if not directory.is_dir():
                continue
            try:
                for path in sorted(directory.rglob("*")):
                    if path.suffix.lower() not in _FONT_SUFFIXES or not path.is_file():
                        continue
                    # First match wins so scan order is deterministic.
                    self._index.setdefault(path.stem.lower(), path)
            except OSError:  # pragma: no cover - unreadable font dir
                logger.debug("could not scan font directory %s", directory, exc_info=True)
        logger.debug("indexed %d fonts", len(self._index))

    def families(self) -> list[str]:
        self.scan()
        return sorted(self._index)

    def resolve(self, family: str | None, size: int) -> tuple[ImageFont.ImageFont, str]:
        """Return ``(font, description)`` for ``family`` at ``size``."""
        self.scan()
        size = max(4, min(int(size), 4096))

        if family:
            path = self._find(family)
            if path is not None:
                try:
                    return ImageFont.truetype(str(path), size), path.stem
                except OSError as exc:
                    logger.warning("font %s could not be loaded: %s", path, exc)

        for candidate in _PREFERRED:
            path = self._index.get(candidate)
            if path is not None:
                try:
                    return ImageFont.truetype(str(path), size), path.stem
                except OSError:  # pragma: no cover - corrupt font file
                    continue

        # No preferred family installed. Rather than dropping to the tiny
        # bitmap default, take whatever real font the image does have - a
        # container with only Adwaita or Noto should still render a usable
        # watermark at the requested size.
        for stem in sorted(self._index):
            try:
                return ImageFont.truetype(str(self._index[stem]), size), stem
            except OSError:  # pragma: no cover - corrupt font file
                continue

        # Pillow's bundled default is scalable when given a size, so text still
        # renders at the requested size in a fontless container.
        try:
            return ImageFont.load_default(size=size), "pillow-default"
        except TypeError:  # pragma: no cover - Pillow < 10.1
            return ImageFont.load_default(), "pillow-default-legacy"

    def _find(self, family: str) -> Path | None:
        key = family.strip().lower().replace(" ", "").replace("-", "")
        if not key:
            return None
        exact = self._index.get(key)
        if exact is not None:
            return exact
        for stem, path in self._index.items():
            if stem.replace("-", "") == key:
                return path
        for stem, path in self._index.items():
            if key in stem.replace("-", ""):
                return path
        return None


__all__ = ["FontIndex"]

"""Shared test fixtures.

Fixtures deliberately build a real sandbox on disk rather than mocking the
filesystem: path confinement, symlink handling and descriptor verification are
exactly the things under test, and a mock would happily accept a path that the
real kernel rejects.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pictor_mcp.config import Config, load_config  # noqa: E402
from pictor_mcp.imaging.fonts import FontIndex  # noqa: E402
from pictor_mcp.imaging.loader import ImageLoader  # noqa: E402
from pictor_mcp.outputs import ResultBuilder  # noqa: E402
from pictor_mcp.security.net import SafeFetcher  # noqa: E402
from pictor_mcp.security.paths import PathJail  # noqa: E402
from pictor_mcp.security.ratelimit import ConcurrencyGate  # noqa: E402


@dataclass(slots=True)
class Sandbox:
    root: Path
    input: Path
    output: Path

    def inputs(self, name: str) -> Path:
        return self.input / name

    def outputs(self, name: str) -> Path:
        return self.output / name


def _photo(width: int = 320, height: int = 240) -> Image.Image:
    """A synthetic image with gradients and shapes, so operations are visible."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    array[:, :, 2] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    image = Image.fromarray(array, "RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse((width * 0.2, height * 0.2, width * 0.5, height * 0.6), fill=(240, 30, 30))
    return image


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    """A confined input/output pair populated with test images."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    _photo().save(input_dir / "photo.jpg", quality=92)
    _photo(160, 120).save(input_dir / "photo.png")
    _photo(200, 200).convert("RGBA").save(input_dir / "alpha.png")

    # A flat white canvas with a blue ellipse: exercises colour keying.
    flat = Image.new("RGB", (200, 150), (250, 250, 250))
    ImageDraw.Draw(flat).ellipse((40, 30, 160, 120), fill=(20, 120, 200))
    flat.save(input_dir / "flat.png")

    logo = Image.new("RGBA", (60, 30), (255, 0, 0, 255))
    logo.save(input_dir / "logo.png")

    # Animation, to exercise multi-frame handling. The frames must actually
    # differ: identical frames are legitimately collapsed by the GIF encoder,
    # which would make this fixture a single-frame file and silently weaken
    # every animation test that uses it.
    frames = []
    for index in range(4):
        frame = _photo(64, 64).convert("RGB")
        ImageDraw.Draw(frame).rectangle((0, 0, 63, 6 + index * 6), fill=(255, 240 - index * 60, index * 50))
        frames.append(frame)
    frames[0].save(
        input_dir / "anim.gif",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )

    # A truncated file: must be rejected, never partially decoded.
    (input_dir / "broken.jpg").write_bytes((input_dir / "photo.jpg").read_bytes()[:128])
    # Not an image at all.
    (input_dir / "notimage.txt").write_bytes(b"this is not an image")

    nested = input_dir / "nested"
    nested.mkdir()
    _photo(80, 80).save(nested / "deep.png")

    return Sandbox(root=tmp_path, input=input_dir, output=output_dir)


def base_env(sandbox: Sandbox, **overrides: str) -> dict[str, str]:
    env = {
        "PICTOR_INPUT_ROOTS": str(sandbox.input),
        "PICTOR_OUTPUT_ROOT": str(sandbox.output),
        "PICTOR_LOG_LEVEL": "WARNING",
    }
    env.update({key: str(value) for key, value in overrides.items()})
    return env


@pytest.fixture
def config(sandbox: Sandbox) -> Config:
    return load_config(base_env(sandbox))


@pytest.fixture
def jail(config: Config) -> PathJail:
    created = PathJail(config.input_roots, config.output_root)
    created.ensure_output_root()
    return created


@pytest.fixture
def loader(config: Config, jail: PathJail) -> ImageLoader:
    return ImageLoader(config, jail, SafeFetcher(config.fetch))


@pytest.fixture
def builder(config: Config, jail: PathJail) -> ResultBuilder:
    return ResultBuilder(config, jail)


@pytest.fixture
def fonts(config: Config) -> FontIndex:
    return FontIndex(config.font_dirs)


@pytest.fixture
def gate(config: Config) -> ConcurrencyGate:
    return ConcurrencyGate(config.limits.max_concurrency)


@pytest.fixture
def server_env(sandbox: Sandbox) -> dict[str, str]:
    """Environment for spawning a real server subprocess over stdio."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(SRC),
        "PYTHONUNBUFFERED": "1",
        "PICTOR_TRANSPORT": "stdio",
        "PICTOR_INPUT_ROOTS": str(sandbox.input),
        "PICTOR_OUTPUT_ROOT": str(sandbox.output),
        "PICTOR_LOG_LEVEL": "ERROR",
        "PICTOR_FONT_DIRS": "/usr/share/fonts",
    }
    return env

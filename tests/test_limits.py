"""Resource-limit tests: decompression bombs, oversized inputs, output ceilings."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from pictor_mcp.config import Limits, load_config
from pictor_mcp.errors import InvalidArgumentError, LimitExceededError, UnsupportedFormatError
from pictor_mcp.imaging.loader import ImageLoader, ImageSource
from pictor_mcp.security.limits import (
    Geometry,
    check_base64_size,
    check_decoded_geometry,
    check_output_geometry,
    configure_pillow,
)

from .conftest import Sandbox, base_env

pytestmark = pytest.mark.anyio


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestGeometryChecks:
    def test_accepts_a_reasonable_image(self) -> None:
        check_decoded_geometry(Geometry(1920, 1080), Limits())

    def test_rejects_a_pixel_count_over_the_limit(self) -> None:
        with pytest.raises(LimitExceededError, match="decompression bomb"):
            check_decoded_geometry(Geometry(20_000, 20_000), Limits(max_pixels=1_000_000))

    def test_rejects_an_overlong_single_axis(self) -> None:
        with pytest.raises(LimitExceededError, match="per-axis"):
            check_decoded_geometry(Geometry(100_000, 10), Limits(max_dimension=10_000))

    def test_rejects_an_oversized_frame_count(self) -> None:
        with pytest.raises(LimitExceededError, match="frame limit"):
            check_decoded_geometry(Geometry(10, 10, frames=5000), Limits(max_frames=100))

    def test_rejects_zero_area(self) -> None:
        with pytest.raises(LimitExceededError):
            check_decoded_geometry(Geometry(0, 10), Limits())

    def test_output_ceiling_is_checked_before_allocating(self) -> None:
        with pytest.raises(LimitExceededError):
            check_output_geometry(200_000, 200_000, Limits(max_output_pixels=10_000_000))

    def test_output_axis_ceiling(self) -> None:
        with pytest.raises(LimitExceededError):
            check_output_geometry(50_000, 10, Limits(max_dimension=10_000))

    def test_base64_cap(self) -> None:
        with pytest.raises(LimitExceededError):
            check_base64_size(b"x" * 5000, Limits(max_input_base64_bytes=1024))


class TestDecodePath:
    async def test_rejects_a_decompression_bomb_before_decoding(self, sandbox: Sandbox, loader: ImageLoader) -> None:
        """A small file that declares enormous dimensions must be refused."""
        bomb = sandbox.inputs("bomb.png")
        # 20000x20000 of a single colour compresses to a few kilobytes.
        Image.new("L", (20000, 20000), 0).save(bomb, optimize=True)
        assert bomb.stat().st_size < 1_000_000

        config = load_config(base_env(sandbox, PICTOR_MAX_PIXELS="1000000"))
        from pictor_mcp.security.net import SafeFetcher
        from pictor_mcp.security.paths import PathJail

        jail = PathJail(config.input_roots, config.output_root)
        restricted = ImageLoader(config, jail, SafeFetcher(config.fetch))
        with pytest.raises(LimitExceededError):
            await restricted.load(ImageSource.parse(path="bomb.png"))

    async def test_rejects_a_truncated_file(self, loader: ImageLoader) -> None:
        with pytest.raises(UnsupportedFormatError):
            await loader.load(ImageSource.parse(path="broken.jpg"))

    async def test_rejects_a_non_image(self, loader: ImageLoader) -> None:
        with pytest.raises(UnsupportedFormatError):
            await loader.load(ImageSource.parse(path="notimage.txt"))

    async def test_rejects_unsupported_inline_base64(self, loader: ImageLoader) -> None:
        with pytest.raises(InvalidArgumentError, match="base64"):
            await loader.load(ImageSource.parse(base64_data="not!valid!base64!"))

    async def test_accepts_a_data_uri(self, loader: ImageLoader) -> None:
        payload = base64.b64encode(_png(8, 8)).decode()
        loaded = await loader.load(ImageSource.parse(base64_data=f"data:image/png;base64,{payload}"))
        assert (loaded.geometry.width, loaded.geometry.height) == (8, 8)

    async def test_accepts_whitespace_in_base64(self, loader: ImageLoader) -> None:
        """Models often pretty-print base64; that should not be a hard failure."""
        payload = base64.b64encode(_png(8, 8)).decode()
        wrapped = "\n".join(payload[i : i + 32] for i in range(0, len(payload), 32))
        loaded = await loader.load(ImageSource.parse(base64_data=wrapped))
        assert loaded.geometry.width == 8

    async def test_rejects_two_inputs_at_once(self, loader: ImageLoader) -> None:
        with pytest.raises(InvalidArgumentError, match="only one"):
            await loader.load(ImageSource.parse(path="photo.jpg", base64_data="AAAA"))

    async def test_rejects_no_input(self) -> None:
        with pytest.raises(InvalidArgumentError, match="exactly one"):
            ImageSource.parse()

    def test_the_decoded_format_decides_not_the_extension(self, sandbox: Sandbox, loader: ImageLoader) -> None:
        """A PNG named .jpg is a PNG - and its extension is never trusted."""
        disguised = sandbox.inputs("actually_png.jpg")
        disguised.write_bytes(_png(16, 16))
        loaded = loader.decode_bytes(
            disguised.read_bytes(),
            source=ImageSource.parse(path="actually_png.jpg"),
            origin="test",
        )
        assert loaded.original_format == "png"


class TestPillowConfiguration:
    def test_pillow_limit_follows_the_config(self) -> None:
        configure_pillow(Limits(max_pixels=12345))
        assert Image.MAX_IMAGE_PIXELS == 12345
        # Restore a sane value for the rest of the session.
        configure_pillow(Limits())

    async def test_a_bomb_is_stopped_by_the_loader_not_only_by_pillow(self, sandbox: Sandbox) -> None:
        """Our own check must fire even if Pillow's warning threshold is raised."""
        bomb = sandbox.inputs("bomb2.png")
        Image.new("L", (6000, 6000), 0).save(bomb)
        config = load_config(base_env(sandbox, PICTOR_MAX_PIXELS="10000"))
        from pictor_mcp.imaging.loader import ImageLoader as Loader
        from pictor_mcp.security.net import SafeFetcher
        from pictor_mcp.security.paths import PathJail

        configure_pillow(config.limits)
        try:
            jail = PathJail(config.input_roots, config.output_root)
            restricted = Loader(config, jail, SafeFetcher(config.fetch))
            with pytest.raises(LimitExceededError):
                await restricted.load(ImageSource.parse(path="bomb2.png"))
        finally:
            configure_pillow(Limits())


class TestAnimationBudget:
    """A per-frame limit and a frame-count limit do not bound their product."""

    def test_a_many_frame_animation_is_rejected(self) -> None:
        # 2000x2000 x 120 frames passes max_pixels and max_frames individually.
        geometry = Geometry(2000, 2000, frames=120)
        assert geometry.pixels < Limits().max_pixels
        assert geometry.frames <= Limits().max_frames
        with pytest.raises(LimitExceededError, match="animation budget"):
            check_decoded_geometry(geometry, Limits())

    def test_a_small_animation_is_allowed(self) -> None:
        check_decoded_geometry(Geometry(800, 600, frames=24), Limits())

    def test_a_still_image_is_not_subject_to_the_animation_budget(self) -> None:
        check_decoded_geometry(Geometry(20_000, 3_000, frames=1), Limits(max_animation_pixels=10_000))

    def test_the_budget_is_configurable(self, sandbox: Sandbox) -> None:
        config = load_config(base_env(sandbox, PICTOR_MAX_ANIMATION_PIXELS="10000"))
        assert config.limits.max_animation_pixels == 10_000

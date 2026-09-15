"""Geometry and pixel-operation tests.

Sizing arithmetic is tested separately from resampling because getting a size
wrong is the most common image bug, and it is pure integer math that deserves
exact assertions.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from pictor_mcp.errors import InvalidArgumentError, LimitExceededError
from pictor_mcp.imaging import ops
from pictor_mcp.imaging.ops import (
    Size,
    auto_orient,
    blur,
    compute_target_size,
    crop_box,
    crop_to_aspect,
    crop_to_size,
    flip,
    parse_gravity,
    resize,
    rotate,
    sharpen,
    trim_border,
)


class TestComputeTargetSize:
    def test_both_dimensions_given(self) -> None:
        assert compute_target_size(Size(100, 50), width=20, height=30) == Size(20, 30)

    def test_width_only_preserves_aspect(self) -> None:
        assert compute_target_size(Size(100, 50), width=20) == Size(20, 10)

    def test_height_only_preserves_aspect(self) -> None:
        assert compute_target_size(Size(100, 50), height=10) == Size(20, 10)

    def test_percent_overrides_dimensions(self) -> None:
        assert compute_target_size(Size(100, 50), width=999, percent=50) == Size(50, 25)

    def test_percent_rounds_to_at_least_one_pixel(self) -> None:
        assert compute_target_size(Size(100, 50), percent=0.5) == Size(1, 1)

    def test_only_shrink_leaves_small_images_alone(self) -> None:
        assert compute_target_size(Size(100, 50), width=500, only_shrink=True) == Size(100, 50)

    def test_only_shrink_still_shrinks(self) -> None:
        assert compute_target_size(Size(100, 50), width=50, only_shrink=True) == Size(50, 25)

    def test_requires_something_to_work_with(self) -> None:
        with pytest.raises(InvalidArgumentError):
            compute_target_size(Size(100, 50))

    def test_rejects_non_positive_percent(self) -> None:
        with pytest.raises(InvalidArgumentError):
            compute_target_size(Size(100, 50), percent=0)

    def test_aspect_ratio_is_preserved_under_rounding(self) -> None:
        """A 3-pixel rounding drift on a 16:9 image should be within a pixel."""
        target = compute_target_size(Size(1920, 1080), width=640)
        assert target.height in {359, 360, 361}


class TestFitModes:
    source = Size(200, 100)

    def _image(self) -> Image.Image:
        return Image.new("RGB", (200, 100), (10, 20, 30))

    def test_contain_fits_inside_and_keeps_ratio(self) -> None:
        result = resize(self._image(), Size(100, 100), fit="contain")
        assert result.size == (100, 50)

    def test_contain_can_upscale(self) -> None:
        result = resize(self._image(), Size(400, 400), fit="contain")
        assert result.size == (400, 200)

    def test_inside_never_upscales(self) -> None:
        result = resize(self._image(), Size(400, 400), fit="inside")
        assert result.size == (200, 100)

    def test_cover_fills_the_box_exactly(self) -> None:
        result = resize(self._image(), Size(100, 100), fit="cover")
        assert result.size == (100, 100)

    def test_fill_stretches_exactly(self) -> None:
        assert resize(self._image(), Size(64, 64), fit="fill").size == (64, 64)

    def test_outside_never_crops(self) -> None:
        """outside scales until the box is covered, and keeps the whole frame."""
        result = resize(self._image(), Size(100, 100), fit="outside")
        assert result.size == (200, 100)
        assert result.width >= 100 and result.height >= 100

    def test_outside_scales_up_for_a_taller_box(self) -> None:
        result = resize(self._image(), Size(100, 300), fit="outside")
        assert result.size == (600, 300)

    def test_pad_produces_the_exact_box(self) -> None:
        result = resize(self._image(), Size(100, 100), fit="pad", background=(255, 0, 0, 255))
        assert result.size == (100, 100)

    def test_pad_fills_the_exposed_area(self) -> None:
        result = resize(self._image(), Size(100, 100), fit="pad", background=(255, 0, 0, 255))
        assert result.convert("RGB").getpixel((2, 2)) == (255, 0, 0)
        # The image itself is centred and untouched.
        assert result.convert("RGB").getpixel((50, 50)) == (10, 20, 30)

    def test_unknown_fit_is_rejected(self) -> None:
        with pytest.raises(InvalidArgumentError):
            resize(self._image(), Size(10, 10), fit="squish")  # type: ignore[arg-type]

    def test_unknown_filter_is_rejected(self) -> None:
        with pytest.raises(InvalidArgumentError):
            resize(self._image(), Size(10, 10), filter_name="magic")

    @pytest.mark.parametrize("name", ["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos", "auto", None])
    def test_every_filter_is_accepted(self, name: str | None) -> None:
        # contain fits inside the box, so only the width is pinned.
        assert resize(self._image(), Size(50, 50), filter_name=name, fit="fill").size == (50, 50)

    def test_auto_filter_differs_by_direction(self) -> None:
        assert ops.resolve_filter(None, upscaling=False) == Image.Resampling.LANCZOS
        assert ops.resolve_filter(None, upscaling=True) == Image.Resampling.BICUBIC

    def test_same_size_is_a_noop(self) -> None:
        source = self._image()
        assert resize(source, Size(200, 100)) is source


class TestGravity:
    def test_centre(self) -> None:
        assert parse_gravity("center") == (0.5, 0.5)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("top", (0.5, 0.0)),
            ("bottom", (0.5, 1.0)),
            ("left", (0.0, 0.5)),
            ("right", (1.0, 0.5)),
            ("top-left", (0.0, 0.0)),
            ("bottom-right", (1.0, 1.0)),
        ],
    )
    def test_named_positions(self, name: str, expected: tuple[float, float]) -> None:
        assert parse_gravity(name) == expected

    def test_underscores_and_case_are_tolerated(self) -> None:
        assert parse_gravity("BOTTOM_RIGHT") == (1.0, 1.0)

    def test_unknown_gravity_is_rejected(self) -> None:
        with pytest.raises(InvalidArgumentError):
            parse_gravity("middle-ish")

    def test_crop_to_size_respects_gravity(self) -> None:
        image = Image.new("RGB", (100, 100))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 99, 49), fill=(255, 0, 0))  # red top half
        top = crop_to_size(image, Size(50, 50), gravity="top")
        bottom = crop_to_size(image, Size(50, 50), gravity="bottom")
        assert top.getpixel((25, 25))[0] > 200
        assert bottom.getpixel((25, 25))[0] < 50


class TestCrop:
    def test_crop_box(self) -> None:
        assert crop_box(Image.new("RGB", (100, 100)), (10, 20, 60, 80)).size == (50, 60)

    def test_crop_box_clamps_by_default(self) -> None:
        assert crop_box(Image.new("RGB", (100, 100)), (-20, -20, 500, 500)).size == (100, 100)

    def test_crop_box_can_refuse_to_clamp(self) -> None:
        with pytest.raises(InvalidArgumentError):
            crop_box(Image.new("RGB", (100, 100)), (0, 0, 500, 500), clamp=False)

    def test_degenerate_box_is_rejected(self) -> None:
        with pytest.raises(InvalidArgumentError):
            crop_box(Image.new("RGB", (100, 100)), (50, 50, 50, 50))

    def test_crop_to_aspect_from_a_wide_image(self) -> None:
        assert crop_to_aspect(Image.new("RGB", (100, 50)), 1.0).size == (50, 50)

    def test_crop_to_aspect_from_a_tall_image(self) -> None:
        assert crop_to_aspect(Image.new("RGB", (50, 100)), 1.0).size == (50, 50)

    def test_crop_to_aspect_rejects_zero(self) -> None:
        with pytest.raises(InvalidArgumentError):
            crop_to_aspect(Image.new("RGB", (50, 50)), 0)

    def test_trim_removes_a_uniform_border(self) -> None:
        image = Image.new("RGB", (100, 100), (255, 255, 255))
        ImageDraw.Draw(image).rectangle((20, 30, 79, 69), fill=(0, 0, 0))
        assert trim_border(image).size == (60, 40)

    def test_trim_tolerates_compression_noise(self) -> None:
        image = Image.new("RGB", (60, 60), (250, 250, 250))
        draw = ImageDraw.Draw(image)
        draw.rectangle((15, 15, 44, 44), fill=(0, 0, 0))
        draw.point((0, 0), fill=(244, 246, 248))
        assert trim_border(image, tolerance=12).size == (30, 30)

    def test_trim_rejects_a_uniform_image(self) -> None:
        with pytest.raises(InvalidArgumentError):
            trim_border(Image.new("RGB", (20, 20), (5, 5, 5)))

    def test_trim_rejects_negative_tolerance(self) -> None:
        with pytest.raises(InvalidArgumentError):
            trim_border(Image.new("RGB", (20, 20)), tolerance=-1)


class TestRotateFlip:
    def test_rotate_by_90_swaps_dimensions_when_expanding(self) -> None:
        assert rotate(Image.new("RGB", (100, 50)), 90, expand=True).size == (50, 100)

    def test_rotate_by_90_without_expanding_keeps_the_canvas(self) -> None:
        assert rotate(Image.new("RGB", (100, 50)), 90, expand=False).size == (100, 50)

    def test_rotate_by_zero_is_a_noop(self) -> None:
        source = Image.new("RGB", (10, 10))
        assert rotate(source, 0) is source

    def test_rotate_by_360_is_a_noop(self) -> None:
        source = Image.new("RGB", (10, 10))
        assert rotate(source, 360) is source

    def test_rotate_fills_with_the_requested_colour(self) -> None:
        result = rotate(Image.new("RGB", (40, 40), (0, 0, 0)), 45, background=(255, 0, 0, 255))
        assert result.convert("RGB").getpixel((1, 1)) == (255, 0, 0)

    def test_flip_horizontal(self) -> None:
        image = Image.new("RGB", (4, 1))
        image.putpixel((0, 0), (255, 0, 0))
        assert flip(image, "horizontal").getpixel((3, 0)) == (255, 0, 0)

    def test_flip_vertical(self) -> None:
        image = Image.new("RGB", (1, 4))
        image.putpixel((0, 0), (255, 0, 0))
        assert flip(image, "vertical").getpixel((0, 3)) == (255, 0, 0)

    def test_flip_both_is_180_degrees(self) -> None:
        image = Image.new("RGB", (4, 4))
        image.putpixel((0, 0), (255, 0, 0))
        assert flip(image, "both").getpixel((3, 3)) == (255, 0, 0)

    def test_unknown_flip_direction_is_rejected(self) -> None:
        with pytest.raises(InvalidArgumentError):
            flip(Image.new("RGB", (4, 4)), "diagonal")


class TestAutoOrient:
    def test_applies_the_exif_orientation(self) -> None:
        """A photo tagged as rotated must come out the way a viewer shows it."""
        image = Image.new("RGB", (40, 20), (0, 0, 255))
        exif = Image.Exif()
        exif[274] = 6  # rotate 90 CW on display
        image.info["exif"] = exif.tobytes()

        oriented = auto_orient(image)
        assert oriented.size == (20, 40)

    def test_untagged_images_are_unchanged(self) -> None:
        image = Image.new("RGB", (40, 20))
        assert auto_orient(image).size == (40, 20)


class TestFilters:
    def test_sharpen_with_zero_amount_is_a_noop(self) -> None:
        image = Image.new("RGB", (8, 8))
        assert sharpen(image, amount=0) is image

    def test_sharpen_changes_a_blurred_edge(self) -> None:
        image = Image.new("L", (32, 32), 0)
        ImageDraw.Draw(image).rectangle((16, 0, 31, 31), fill=128)
        blurred = blur(image, radius=2)
        assert list(sharpen(blurred, amount=2).getdata()) != list(blurred.getdata())

    def test_blur_rejects_zero_radius(self) -> None:
        with pytest.raises(InvalidArgumentError):
            blur(Image.new("L", (8, 8)), radius=0)

    def test_sharpen_rejects_a_bad_threshold(self) -> None:
        with pytest.raises(InvalidArgumentError):
            sharpen(Image.new("L", (8, 8)), threshold=999)


class TestOutputGuard:
    def test_enforce_output_limits_rejects_an_oversized_result(self) -> None:
        from pictor_mcp.config import Limits

        with pytest.raises(LimitExceededError):
            ops.enforce_output_limits(50_000, 50_000, Limits(max_output_pixels=1_000_000))

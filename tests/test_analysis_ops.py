"""Pipeline, smart-crop, comparison and background-removal tests."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw
from pydantic import TypeAdapter, ValidationError

from pictor_mcp.errors import InvalidArgumentError
from pictor_mcp.imaging import background, compare, smartcrop
from pictor_mcp.imaging.fonts import FontIndex
from pictor_mcp.imaging.ops import Size
from pictor_mcp.imaging.pipeline import (
    AutoOrientOperation,
    BackgroundRemoveOperation,
    CropOperation,
    Operation,
    ResizeOperation,
    RotateOperation,
    SharpenOperation,
    SmartCropOperation,
    WatermarkImageOperation,
    WatermarkTextOperation,
    apply_operations,
    describe_operations,
    watermark_paths,
)


def _run(image: Image.Image, operations: list[Operation], config, fonts=None, overlays=None):
    return apply_operations(
        image,
        operations,
        config=config,
        fonts=fonts or FontIndex(()),
        overlays=overlays or {},
    )


class TestPipeline:
    def test_runs_operations_in_order(self, config) -> None:
        image = Image.new("RGB", (200, 100), (10, 20, 30))
        outcome = _run(
            image,
            [
                ResizeOperation(width=100),
                CropOperation(box=(0, 0, 50, 25)),
            ],
            config,
        )
        assert outcome.image.size == (50, 25)
        assert outcome.steps == ["0:resize", "1:crop"]

    def test_empty_operation_list_is_allowed_and_is_a_noop(self, config) -> None:
        image = Image.new("RGB", (10, 10))
        assert _run(image, [], config).image.size == (10, 10)

    def test_watermark_text_is_applied(self, config) -> None:
        image = Image.new("RGB", (200, 100), (0, 0, 0))
        outcome = _run(image, [WatermarkTextOperation(text="X", font_size=40)], config)
        assert outcome.image.size == (200, 100)
        assert any("watermark" in note for note in outcome.notes)

    def test_missing_watermark_overlay_is_an_error(self, config) -> None:
        image = Image.new("RGB", (50, 50))
        with pytest.raises(InvalidArgumentError, match="pre-loaded"):
            _run(image, [WatermarkImageOperation(path="logo.png")], config)

    def test_watermark_overlay_is_composited(self, config) -> None:
        image = Image.new("RGB", (100, 100), (255, 255, 255))
        overlay = Image.new("RGBA", (20, 20), (255, 0, 0, 255))
        outcome = _run(
            image,
            [WatermarkImageOperation(path="logo.png", position="top-left", opacity=1.0, padding=0)],
            config,
            overlays={"logo.png": overlay},
        )
        assert outcome.image.convert("RGB").getpixel((5, 5)) == (255, 0, 0)

    def test_watermark_paths_are_collected_for_preloading(self) -> None:
        operations: list[Operation] = [
            WatermarkImageOperation(path="a.png"),
            ResizeOperation(width=10),
            WatermarkImageOperation(path="b.png"),
        ]
        assert watermark_paths(operations) == ["a.png", "b.png"]

    def test_oversized_intermediate_result_is_rejected(self, config) -> None:
        """The ceiling is enforced after every step, not only at the end."""
        from pictor_mcp.config import load_config
        from pictor_mcp.errors import LimitExceededError

        tiny = load_config(
            {
                "PICTOR_INPUT_ROOTS": str(config.input_roots[0]),
                "PICTOR_OUTPUT_ROOT": str(config.output_root),
                "PICTOR_MAX_OUTPUT_PIXELS": "10000",
            }
        )
        with pytest.raises(LimitExceededError):
            _run(Image.new("RGB", (50, 50)), [ResizeOperation(width=1000, height=1000)], tiny)

    def test_failing_step_reports_its_index(self, config) -> None:
        image = Image.new("RGB", (50, 50))
        with pytest.raises(InvalidArgumentError, match=r"step 1 \(crop\)"):
            _run(
                image,
                [ResizeOperation(width=10), CropOperation(box=(9, 9, 9, 9))],
                config,
            )

    def test_crop_without_parameters_is_rejected(self, config) -> None:
        with pytest.raises(InvalidArgumentError, match="box, aspect_ratio or trim"):
            _run(Image.new("RGB", (10, 10)), [CropOperation()], config)

    def test_auto_orient_then_rotate_is_cumulative(self, config) -> None:
        image = Image.new("RGB", (40, 20))
        exif = Image.Exif()
        exif[274] = 6
        image.info["exif"] = exif.tobytes()
        outcome = _run(image, [AutoOrientOperation(), RotateOperation(angle=90)], config)
        assert outcome.image.size == (40, 20)

    def test_operations_are_validated_by_the_schema(self) -> None:
        adapter = TypeAdapter(Operation)
        with pytest.raises(ValidationError):
            adapter.validate_python({"op": "resize", "width": -5})
        with pytest.raises(ValidationError):
            adapter.validate_python({"op": "not_a_real_op"})

    def test_extra_fields_are_rejected(self) -> None:
        """A typo in a parameter name must fail, not be silently ignored."""
        adapter = TypeAdapter(Operation)
        with pytest.raises(ValidationError):
            adapter.validate_python({"op": "resize", "wdith": 100})

    def test_describe_operations_lists_every_op(self) -> None:
        described = describe_operations()
        names = {item["op"] for item in described}
        assert {"resize", "crop", "watermark_text", "background_remove"} <= names
        assert all(item["parameters"] for item in described)

    @pytest.mark.parametrize(
        "operation",
        [
            ResizeOperation(width=50),
            RotateOperation(angle=90),
            SharpenOperation(amount=1),
            SmartCropOperation(width=20, height=20),
            CropOperation(aspect_ratio=1.0),
        ],
    )
    def test_each_operation_produces_a_valid_image(self, config, operation: Operation) -> None:
        outcome = _run(Image.new("RGB", (80, 60), (1, 2, 3)), [operation], config)
        assert outcome.image.width > 0 and outcome.image.height > 0


class TestSmartCrop:
    def _subject_on_the_left(self) -> Image.Image:
        """A busy left half and a flat right half: the crop must pick the left."""
        image = Image.new("L", (400, 200), 128)
        draw = ImageDraw.Draw(image)
        for x in range(0, 180, 4):
            draw.line((x, 0, x, 199), fill=255 if (x // 4) % 2 else 0)
        return image.convert("RGB")

    def test_finds_the_busy_region(self) -> None:
        box = smartcrop.best_window(self._subject_on_the_left(), Size(100, 100))
        assert box[0] < 100, f"expected the busy left side, got {box}"

    def test_window_stays_inside_the_image(self) -> None:
        image = self._subject_on_the_left()
        for size in (Size(50, 50), Size(120, 90), Size(400, 200)):
            left, top, right, bottom = smartcrop.best_window(image, size)
            assert 0 <= left < right <= image.width
            assert 0 <= top < bottom <= image.height
            assert right - left == size.width
            assert bottom - top == size.height

    def test_oversized_target_is_clamped_to_the_image(self) -> None:
        image = Image.new("RGB", (40, 40))
        assert smartcrop.best_window(image, Size(100, 100)) == (0, 0, 40, 40)

    def test_centre_method_centres_on_a_symmetric_image(self) -> None:
        image = Image.new("RGB", (200, 200), (10, 10, 10))
        box = smartcrop.best_window(image, Size(50, 50), method="center")
        assert box == (75, 75, 125, 125)

    def test_uniform_energy_does_not_drift_to_a_corner(self) -> None:
        """A flat image has no best window; the centre bias keeps it predictable."""
        image = Image.new("RGB", (300, 300), (7, 7, 7))
        box = smartcrop.best_window(image, Size(100, 100))
        assert abs(box[0] - 100) <= 20 and abs(box[1] - 100) <= 20

    def test_smart_thumbnail_hits_the_exact_target(self) -> None:
        result = smartcrop.smart_thumbnail(self._subject_on_the_left(), Size(64, 64))
        assert result.size == (64, 64)

    def test_focal_point_is_normalised(self) -> None:
        x, y = smartcrop.focal_point(self._subject_on_the_left())
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        assert x < 0.5, "the subject is on the left"

    def test_unknown_method_is_rejected(self) -> None:
        with pytest.raises(InvalidArgumentError):
            smartcrop.saliency_map(Image.new("RGB", (10, 10)), "vibes")

    def test_entropy_method_runs(self) -> None:
        assert smartcrop.saliency_map(Image.new("RGB", (60, 60)), "entropy").shape == (60, 60)


class TestCompare:
    def test_identical_images_score_perfectly(self) -> None:
        image = Image.new("RGB", (64, 64), (10, 20, 30))
        result = compare.compare_images(image, image.copy())
        assert result.identical is True
        assert result.ssim == pytest.approx(1.0)
        assert result.rmse == 0
        assert result.hash_hamming_distance == 0

    def test_different_images_score_lower(self) -> None:
        a = Image.new("RGB", (64, 64), (0, 0, 0))
        b = Image.new("RGB", (64, 64), (255, 255, 255))
        result = compare.compare_images(a, b)
        assert result.identical is False
        assert result.ssim < 0.5
        assert result.max_difference == 255

    def test_align_resizes_the_second_image(self) -> None:
        a = Image.new("RGB", (100, 100), (5, 5, 5))
        b = Image.new("RGB", (50, 50), (5, 5, 5))
        result = compare.compare_images(a, b, align=True)
        assert result.same_size is False
        assert result.width_b == 50 and result.height_b == 50
        assert result.ssim > 0.99

    def test_mismatched_sizes_without_align_is_an_error(self) -> None:
        with pytest.raises(InvalidArgumentError, match="differ in size"):
            compare.compare_images(Image.new("RGB", (10, 10)), Image.new("RGB", (20, 20)), align=False)

    def test_a_small_shift_barely_moves_ssim_but_moves_rmse_a_lot(self) -> None:
        """SSIM is the metric that matches perception; RMSE is not."""
        base = Image.new("L", (64, 64), 0)
        draw = ImageDraw.Draw(base)
        draw.rectangle((20, 20, 40, 40), fill=200)
        shifted = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(shifted).rectangle((21, 20, 41, 40), fill=200)

        one = compare.compare_arrays(np.asarray(base, dtype=float), np.asarray(shifted, dtype=float))
        assert one.ssim > 0.8
        assert one.rmse > 10

    def test_changed_pixel_ratio_respects_the_threshold(self) -> None:
        a = Image.new("L", (10, 10), 100)
        b = Image.new("L", (10, 10), 104)
        assert (
            compare.compare_arrays(np.asarray(a, float), np.asarray(b, float), change_threshold=8).changed_pixel_ratio
            == 0
        )
        assert (
            compare.compare_arrays(np.asarray(a, float), np.asarray(b, float), change_threshold=2).changed_pixel_ratio
            == 1.0
        )

    def test_hashes_are_stable_across_a_resize(self) -> None:
        """Perceptual hashes must survive re-encoding, which is their whole point.

        dHash is bit-exact across a 2x resize; pHash shifts by a few bits
        because the DCT is sensitive to resampling of a hard edge. A small
        Hamming distance is the property that makes near-duplicate detection
        work, so that is what is asserted - exact equality would be a stricter
        claim than the algorithm supports.
        """
        array = np.zeros((256, 256, 3), np.uint8)
        array[:, :128] = 255
        image = Image.fromarray(array, "RGB")
        small = image.resize((128, 128))
        first = compare.perceptual_hashes(image)
        second = compare.perceptual_hashes(small)
        assert first["dhash"] == second["dhash"]
        assert first["ahash"] == second["ahash"]
        distance = bin(int(first["phash"], 16) ^ int(second["phash"], 16)).count("1")
        assert distance <= 10, f"pHash drifted by {distance} bits across a resize"

    def test_hash_similarity_is_reported(self) -> None:
        array = np.zeros((64, 64, 3), np.uint8)
        array[:, :32] = 255
        image = Image.fromarray(array, "RGB")
        result = compare.compare_images(image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
        assert result.hash_hamming_distance > 0
        assert 0.0 <= result.hash_similarity <= 1.0

    def test_difference_image_highlights_changes(self) -> None:
        a = Image.new("RGB", (16, 16), (0, 0, 0))
        b = Image.new("RGB", (16, 16), (0, 0, 0))
        ImageDraw.Draw(b).rectangle((4, 4, 8, 8), fill=(255, 255, 255))
        diff = compare.difference_image(a, b).convert("L")
        assert diff.getpixel((6, 6)) == 255
        assert diff.getpixel((0, 0)) == 0

    def test_ssim_requires_matching_shapes(self) -> None:
        with pytest.raises(InvalidArgumentError):
            compare.ssim(np.zeros((4, 4)), np.zeros((8, 8)))


class TestBackgroundRemoval:
    def _flat_subject(self) -> Image.Image:
        image = Image.new("RGB", (100, 100), (250, 250, 250))
        ImageDraw.Draw(image).ellipse((30, 30, 70, 70), fill=(20, 120, 200))
        return image

    def test_colour_key_removes_the_border(self) -> None:
        result, notes = background.remove_background_color(self._flat_subject(), tolerance=30)
        alpha = np.asarray(result)[:, :, 3]
        assert alpha[0, 0] == 0, "the flat background should be transparent"
        assert alpha[50, 50] == 255, "the subject should stay opaque"
        assert any("border-connected" in note for note in notes)

    def test_interior_matching_colour_is_preserved_when_edge_connected(self) -> None:
        """A white shirt on a white background must keep its shirt."""
        image = Image.new("RGB", (100, 100), (250, 250, 250))
        # An enclosed white circle with a coloured ring around it.
        draw = ImageDraw.Draw(image)
        draw.ellipse((20, 20, 80, 80), fill=(20, 120, 200))
        draw.ellipse((40, 40, 60, 60), fill=(250, 250, 250))

        result, _ = background.remove_background_color(image, tolerance=30, edge_connected=True)
        alpha = np.asarray(result)[:, :, 3]
        assert alpha[0, 0] == 0
        assert alpha[50, 50] == 255, "the enclosed white region must survive"

    def test_without_edge_connectivity_the_interior_is_removed(self) -> None:
        image = Image.new("RGB", (100, 100), (250, 250, 250))
        draw = ImageDraw.Draw(image)
        draw.ellipse((20, 20, 80, 80), fill=(20, 120, 200))
        draw.ellipse((40, 40, 60, 60), fill=(250, 250, 250))

        result, _ = background.remove_background_color(image, tolerance=30, edge_connected=False)
        assert np.asarray(result)[50, 50, 3] == 0

    def test_softness_produces_intermediate_alpha(self) -> None:
        image = Image.new("RGB", (20, 20), (255, 255, 255))
        ImageDraw.Draw(image).rectangle((8, 8, 11, 11), fill=(200, 200, 200))
        hard, _ = background.remove_background_color(image, tolerance=30, softness=0)
        soft, _ = background.remove_background_color(image, tolerance=30, softness=80)
        assert len(np.unique(np.asarray(soft)[:, :, 3])) >= len(np.unique(np.asarray(hard)[:, :, 3]))

    def test_compositing_onto_a_colour(self) -> None:
        result, notes = background.remove_background_color(
            self._flat_subject(), tolerance=30, background=(0, 0, 255, 255)
        )
        assert np.asarray(result)[0, 0, 3] == 255
        assert tuple(np.asarray(result)[0, 0][:3]) == (0, 0, 255)
        assert any("composited" in note for note in notes)

    def test_ml_method_reports_unavailability_clearly(self, config) -> None:
        """Without the optional dependency the message must be actionable."""
        from pictor_mcp.errors import BackendUnavailableError

        available, _detail = background.ml_available()
        if available:
            pytest.skip("rembg is installed in this environment")
        with pytest.raises(BackendUnavailableError) as excinfo:
            background.remove_background_ml(Image.new("RGB", (8, 8)))
        assert "rembg" in str(excinfo.value) or "onnxruntime" in str(excinfo.value)

    def test_auto_falls_back_to_colour_keying(self) -> None:
        result, notes = background.remove_background(self._flat_subject(), method="auto")
        assert np.asarray(result)[0, 0, 3] == 0
        assert notes

    def test_unknown_method_is_rejected(self) -> None:
        with pytest.raises(InvalidArgumentError):
            background.remove_background(Image.new("RGB", (8, 8)), method="magic")

    def test_feather_softens_the_alpha_edge(self) -> None:
        image = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        ImageDraw.Draw(image).rectangle((10, 10, 30, 30), fill=(255, 0, 0, 255))
        feathered = background.feather_alpha(image, 3)
        assert len(np.unique(np.asarray(feathered)[:, :, 3])) > 2


class TestBackgroundInPipeline:
    def test_pipeline_background_removal(self, config) -> None:
        image = Image.new("RGB", (60, 60), (250, 250, 250))
        ImageDraw.Draw(image).ellipse((15, 15, 45, 45), fill=(10, 10, 10))
        outcome = _run(
            image,
            [BackgroundRemoveOperation(method="color", tolerance=30)],
            config,
        )
        assert outcome.image.mode == "RGBA"
        assert np.asarray(outcome.image)[0, 0, 3] == 0
        assert any("keyed out" in note for note in outcome.notes)

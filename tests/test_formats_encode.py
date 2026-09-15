"""Format allow-list and encoder tests."""

from __future__ import annotations

import pytest
from PIL import Image

from pictor_mcp.errors import InvalidArgumentError, UnsupportedFormatError
from pictor_mcp.imaging.encode import EncodeOptions, encode, encode_animation, flatten_alpha
from pictor_mcp.imaging.formats import (
    INPUT_FORMATS,
    OUTPUT_FORMATS,
    assert_decoded_format_allowed,
    public_format_catalogue,
    resolve_input_format,
    resolve_output_format,
)


class TestFormatRegistry:
    def test_aliases_resolve(self) -> None:
        assert resolve_output_format("jpg").key == "jpeg"
        assert resolve_output_format(".JPEG").key == "jpeg"
        assert resolve_output_format("tif").key == "tiff"
        assert resolve_output_format("image/webp").key == "webp"

    def test_unknown_format_is_rejected_with_alternatives(self) -> None:
        with pytest.raises(UnsupportedFormatError) as excinfo:
            resolve_output_format("heic")
        assert "supported" in excinfo.value.details

    @pytest.mark.parametrize("name", ["pdf", "eps", "ps", "wmf", "hdf5", "grib", "fits", "svg"])
    def test_document_and_scientific_formats_are_not_writable(self, name: str) -> None:
        """An image converter that writes PDF is a document-forgery primitive."""
        with pytest.raises(UnsupportedFormatError):
            resolve_output_format(name)

    @pytest.mark.parametrize("name", ["pdf", "eps", "ps", "wmf", "hdf5", "grib", "fits"])
    def test_document_and_scientific_formats_are_not_readable(self, name: str) -> None:
        with pytest.raises(UnsupportedFormatError):
            resolve_input_format(name)

    def test_decoded_format_is_validated(self) -> None:
        assert assert_decoded_format_allowed("PNG").key == "png"
        with pytest.raises(UnsupportedFormatError):
            assert_decoded_format_allowed("EPS")
        with pytest.raises(UnsupportedFormatError):
            assert_decoded_format_allowed(None)

    def test_catalogue_only_lists_working_codecs(self) -> None:
        catalogue = public_format_catalogue()
        assert {item["format"] for item in catalogue["writable"]} >= {"jpeg", "png", "webp"}
        assert all(item["format"] in OUTPUT_FORMATS for item in catalogue["writable"])
        assert all("mime" in item for item in catalogue["readable"])

    def test_psd_is_readable_but_not_writable(self) -> None:
        assert "psd" in INPUT_FORMATS
        assert "psd" not in OUTPUT_FORMATS


class TestEncode:
    # Parametrised over what this Pillow build can really encode, not over the
    # registry: AVIF, JPEG 2000 and QOI depend on build-time libraries, and a
    # stock CI runner need not have all of them. test_core_codecs_are_available
    # pins the ones that must always be there.
    @pytest.mark.parametrize(
        "fmt",
        sorted({item["format"] for item in public_format_catalogue()["writable"]} - {"ico"}),
    )
    def test_round_trips_every_writable_format(self, fmt: str) -> None:
        source = Image.new("RGB", (48, 32), (200, 40, 40))
        encoded = encode(source, resolve_output_format(fmt), EncodeOptions(quality=70))
        assert encoded.data
        reopened = Image.open(__import__("io").BytesIO(encoded.data))
        assert (reopened.width, reopened.height) == (48, 32)

    def test_core_codecs_are_available(self) -> None:
        """These three must exist in any environment the server claims to support."""
        writable = {item["format"] for item in public_format_catalogue()["writable"]}
        assert {"jpeg", "png", "webp"} <= writable

    def test_ico_produces_a_square_icon(self) -> None:
        """ICO has no concept of a non-square image; Pillow fits the largest square."""
        source = Image.new("RGB", (48, 32), (200, 40, 40))
        encoded = encode(source, resolve_output_format("ico"), EncodeOptions())
        reopened = Image.open(__import__("io").BytesIO(encoded.data))
        assert reopened.width == reopened.height

    def test_jpeg_at_high_quality_survives_an_encoder_quirk(self) -> None:
        """Regression: 4:4:4 chroma plus optimize fails in some libjpeg builds.

        The server must degrade to a slightly larger file rather than failing a
        request that is entirely reasonable.
        """
        import numpy as np

        array = np.random.default_rng(11).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        source = Image.fromarray(array, "RGB")
        encoded = encode(source, resolve_output_format("jpeg"), EncodeOptions(quality=95))
        assert encoded.data.startswith(b"\xff\xd8")

    def test_alpha_is_flattened_for_a_codec_without_transparency(self) -> None:
        """A screenshot on a transparent background must not silently go black."""
        source = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        encoded = encode(source, resolve_output_format("jpeg"), EncodeOptions(background=(255, 255, 255)))
        assert encoded.mode == "RGB"
        assert any("flattened transparency" in note for note in encoded.notes)
        reopened = Image.open(__import__("io").BytesIO(encoded.data)).convert("RGB")
        assert reopened.getpixel((8, 8)) == (255, 255, 255)

    def test_alpha_survives_for_a_codec_with_transparency(self) -> None:
        source = Image.new("RGBA", (16, 16), (10, 20, 30, 128))
        encoded = encode(source, resolve_output_format("png"))
        reopened = Image.open(__import__("io").BytesIO(encoded.data))
        assert reopened.mode == "RGBA"
        assert reopened.getpixel((8, 8))[3] == 128

    def test_cmyk_is_converted_for_rgb_only_codecs(self) -> None:
        source = Image.new("CMYK", (16, 16), (0, 200, 200, 0))
        encoded = encode(source, resolve_output_format("webp"))
        assert encoded.mode in {"RGB", "RGBA"}

    def test_quality_changes_the_size(self) -> None:
        source = Image.new("RGB", (256, 256))
        import numpy as np

        array = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        source = Image.fromarray(array, "RGB")
        spec = resolve_output_format("jpeg")
        low = encode(source, spec, EncodeOptions(quality=10)).byte_size
        high = encode(source, spec, EncodeOptions(quality=95)).byte_size
        assert low < high

    def test_lossless_webp_beats_lossy_on_size(self) -> None:
        import numpy as np

        array = np.random.default_rng(3).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        source = Image.fromarray(array, "RGB")
        spec = resolve_output_format("webp")
        lossy = encode(source, spec, EncodeOptions(quality=50)).byte_size
        lossless = encode(source, spec, EncodeOptions(lossless=True)).byte_size
        assert lossless > lossy

    @pytest.mark.parametrize(
        ("key", "value"), [("quality", 0), ("quality", 101), ("compress_level", 12), ("effort", 30)]
    )
    def test_invalid_options_are_rejected(self, key: str, value: int) -> None:
        with pytest.raises(InvalidArgumentError):
            encode(Image.new("RGB", (8, 8)), resolve_output_format("png"), EncodeOptions(**{key: value}))

    def test_invalid_background_channel_is_rejected(self) -> None:
        with pytest.raises(InvalidArgumentError):
            encode(Image.new("RGB", (8, 8)), resolve_output_format("jpeg"), EncodeOptions(background=(300, 0, 0)))


class TestMetadataHygiene:
    def _source_with_metadata(self) -> Image.Image:
        image = Image.new("RGB", (32, 32), (10, 20, 30))
        exif = Image.Exif()
        exif[271] = "TestCamera"  # Make
        exif[34853] = {1: "N"}  # GPS IFD marker
        image.info["exif"] = exif.tobytes()
        image.info["icc_profile"] = b"\x00" * 128
        return image

    def test_metadata_is_stripped_by_default(self) -> None:
        """A resized photo must not still carry the original GPS coordinates."""
        source = self._source_with_metadata()
        encoded = encode(source, resolve_output_format("jpeg"), EncodeOptions(strip_metadata=True))
        reopened = Image.open(__import__("io").BytesIO(encoded.data))
        assert not reopened.info.get("exif")
        assert not reopened.info.get("icc_profile")
        # The source must be untouched, so later steps still see its metadata.
        assert source.info.get("exif")

    def test_metadata_can_be_kept_explicitly(self) -> None:
        source = self._source_with_metadata()
        encoded = encode(source, resolve_output_format("jpeg"), EncodeOptions(strip_metadata=False))
        reopened = Image.open(__import__("io").BytesIO(encoded.data))
        assert reopened.info.get("exif")

    def test_icc_can_be_kept_while_other_metadata_is_stripped(self) -> None:
        source = self._source_with_metadata()
        encoded = encode(
            source,
            resolve_output_format("jpeg"),
            EncodeOptions(strip_metadata=True, keep_icc=True),
        )
        reopened = Image.open(__import__("io").BytesIO(encoded.data))
        assert reopened.info.get("icc_profile")
        assert not reopened.info.get("exif")

    def test_transparency_is_not_treated_as_metadata(self) -> None:
        """Stripping must never drop keys that change the pixels."""
        source = Image.new("P", (16, 16), 5)
        source.info["transparency"] = 5
        encoded = encode(source, resolve_output_format("png"), EncodeOptions(strip_metadata=True))
        reopened = Image.open(__import__("io").BytesIO(encoded.data))
        assert reopened.mode == "P"
        # Pillow may serialise the transparent index as an int or as bytes
        # depending on the PNG path; either way the key must survive.
        assert reopened.info.get("transparency") is not None


class TestFlattenAlpha:
    def test_composites_onto_the_requested_colour(self) -> None:
        source = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
        assert flatten_alpha(source, (0, 0, 255)).getpixel((4, 4)) == (0, 0, 255)

    def test_respects_partial_alpha(self) -> None:
        source = Image.new("RGBA", (8, 8), (255, 255, 255, 128))
        red, green, blue = flatten_alpha(source, (0, 0, 0)).getpixel((4, 4))
        assert 100 < red < 155 and red == green == blue

    def test_leaves_opaque_images_alone(self) -> None:
        source = Image.new("RGB", (8, 8), (1, 2, 3))
        assert flatten_alpha(source, (255, 255, 255)).getpixel((4, 4)) == (1, 2, 3)


class TestAnimation:
    def _frames(self, count: int = 4) -> list[Image.Image]:
        return [Image.new("RGB", (32, 32), (index * 40, 10, 10)) for index in range(count)]

    def test_gif_animation_is_preserved(self) -> None:
        encoded = encode_animation(
            self._frames(),
            resolve_output_format("gif"),
            EncodeOptions(),
            durations=[80] * 4,
            loop=0,
        )
        reopened = Image.open(__import__("io").BytesIO(encoded.data))
        assert reopened.n_frames == 4
        assert reopened.info.get("duration") == 80

    def test_webp_animation_is_preserved(self) -> None:
        encoded = encode_animation(
            self._frames(3),
            resolve_output_format("webp"),
            EncodeOptions(quality=70),
            durations=[100] * 3,
            loop=0,
        )
        reopened = Image.open(__import__("io").BytesIO(encoded.data))
        assert getattr(reopened, "n_frames", 1) == 3

    def test_distinct_frames_are_not_collapsed(self) -> None:
        """Regression: quantising against the first frame merged every frame.

        A palette built from one flat colour maps all later frames to the same
        index, the frames become identical, and the codec legitimately emits a
        single-frame GIF.
        """
        frames = self._frames(4)
        encoded = encode_animation(frames, resolve_output_format("gif"), EncodeOptions(), durations=[80] * 4, loop=0)
        reopened = Image.open(__import__("io").BytesIO(encoded.data))
        assert reopened.n_frames == 4
        seen = set()
        for index in range(4):
            reopened.seek(index)
            seen.add(reopened.convert("RGB").getpixel((16, 16)))
        assert len(seen) == 4, f"frames collapsed to {len(seen)} distinct colours"

    def test_static_format_refuses_animation(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            encode_animation(self._frames(), resolve_output_format("png"), EncodeOptions(), durations=None, loop=0)

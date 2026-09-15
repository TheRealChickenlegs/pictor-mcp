"""Regression tests for the issues found in the adversarial security review.

Each test here corresponds to a specific reported defect. They are written to
fail loudly if the control is weakened again, because every one of them was a
case where a guard existed but a code path went around it.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pictor_mcp.config import Limits, load_config
from pictor_mcp.errors import InvalidArgumentError, LimitExceededError
from pictor_mcp.imaging import compare, ops, smartcrop
from pictor_mcp.imaging.encode import EncodeOptions, encode
from pictor_mcp.imaging.formats import resolve_output_format
from pictor_mcp.imaging.loader import ImageLoader, ImageSource
from pictor_mcp.security.ratelimit import Deadline

from .conftest import Sandbox, base_env

pytestmark = pytest.mark.anyio


class TestOutputLimitsCannotBeBypassed:
    """Every allocation path must enforce the output geometry."""

    TIGHT = Limits(max_output_pixels=10_000, max_dimension=500)

    def _source(self) -> Image.Image:
        return Image.new("RGB", (100, 100), (1, 2, 3))

    @pytest.mark.parametrize("fit", ["cover", "outside"])
    def test_resize_bounds_the_intermediate_allocation(self, fit: str) -> None:
        """A wide, short target scales up hugely before cropping back down.

        `cover` used to allocate the scaled bitmap with no check at all: a
        24000x1 target on a square source produced a 24000x24000 intermediate
        (~2.3 GB) while the cropped result passed the final check.
        """
        with pytest.raises(LimitExceededError):
            ops.resize(self._source(), ops.Size(300, 1), fit=fit, limits=self.TIGHT)

    @pytest.mark.parametrize("fit", ["contain", "pad", "fill", "inside"])
    def test_fit_modes_that_do_not_upscale_the_frame_still_work(self, fit: str) -> None:
        """Only the covering modes allocate beyond the target box."""
        result = ops.resize(self._source(), ops.Size(60, 20), fit=fit, limits=self.TIGHT)
        assert result.width * result.height <= self.TIGHT.max_output_pixels

    @pytest.mark.parametrize("size", [(300, 1), (1, 300)])
    def test_smart_thumbnail_bounds_the_intermediate(self, size: tuple[int, int]) -> None:
        with pytest.raises(LimitExceededError):
            smartcrop.smart_thumbnail(self._source(), ops.Size(*size), limits=self.TIGHT)

    def test_a_within_limits_resize_still_works(self) -> None:
        result = ops.resize(self._source(), ops.Size(50, 50), fit="cover", limits=self.TIGHT)
        assert result.size == (50, 50)

    def test_pipeline_enforces_limits_through_the_resize_operation(self, sandbox: Sandbox) -> None:
        from pictor_mcp.imaging.pipeline import ResizeOperation, apply_operations

        tight = load_config(base_env(sandbox, PICTOR_MAX_OUTPUT_PIXELS="10000", PICTOR_MAX_DIMENSION="500"))
        from pictor_mcp.imaging.fonts import FontIndex

        with pytest.raises(LimitExceededError):
            apply_operations(
                self._source(),
                [ResizeOperation(width=300, height=1, fit="cover")],
                config=tight,
                fonts=FontIndex(()),
            )

    async def test_optimize_web_rejects_an_oversized_width(self, sandbox: Sandbox) -> None:
        """This path encodes directly, so it must enforce the limit itself."""
        from pictor_mcp.backends import BackendRegistry
        from pictor_mcp.imaging.fonts import FontIndex
        from pictor_mcp.outputs import ResultBuilder
        from pictor_mcp.security.net import SafeFetcher
        from pictor_mcp.security.paths import PathJail
        from pictor_mcp.tools.analysis import _encode_variant
        from pictor_mcp.tools.context import ToolContext

        config = load_config(base_env(sandbox, PICTOR_MAX_OUTPUT_PIXELS="10000", PICTOR_MAX_DIMENSION="500"))
        jail = PathJail(config.input_roots, config.output_root)
        jail.ensure_output_root()
        ctx = ToolContext(
            config=config,
            jail=jail,
            loader=ImageLoader(config, jail, SafeFetcher(config.fetch)),
            builder=ResultBuilder(config, jail),
            fonts=FontIndex(()),
            registry=BackendRegistry(),
            gate=__import__("pictor_mcp.security.ratelimit", fromlist=["ConcurrencyGate"]).ConcurrencyGate(1),
        )
        loaded = await ctx.load_input(path="photo.jpg")
        try:
            with pytest.raises(LimitExceededError):
                await _encode_variant(
                    ctx,
                    loaded,
                    resolve_output_format("webp"),
                    target=ops.Size(400, 400),
                    quality=80,
                    max_bytes=None,
                    strip_metadata=True,
                    animated=False,
                )
        finally:
            loaded.close()

    def test_the_widths_schema_is_bounded(self) -> None:
        """An unbounded width list lets a model request a billion-pixel variant."""
        import asyncio

        from pictor_mcp.server import build_context, build_server

        config = load_config({"PICTOR_INPUT_ROOTS": "/tmp/x", "PICTOR_OUTPUT_ROOT": "/tmp/y"})
        server = build_server(config, build_context(config))
        tools = asyncio.run(server.list_tools())
        schema = next(t for t in tools if t.name == "image_optimize_web").input_schema
        item = schema["properties"]["widths"]["anyOf"][0]["items"]
        assert item["maximum"] == 100_000
        assert item["minimum"] == 1

    def test_an_absurd_width_is_rejected_before_encoding(self) -> None:
        from pictor_mcp.tools.analysis import _width_ladder

        with pytest.raises(InvalidArgumentError, match="exceeds"):
            _width_ladder([10_000_000], 1000)


class TestComparisonIsBoundedAndCorrect:
    def test_ssim_is_near_one_for_nearly_identical_high_contrast_images(self) -> None:
        """Regression: catastrophic cancellation made SSIM report -1.0.

        The variance terms are a difference of means, which goes slightly
        negative on high-contrast regions. Dividing by that negative
        denominator produced a large negative score, i.e. "maximally different"
        for two images that differ in 0.06% of pixels.
        """
        a = np.zeros((600, 600), np.uint8)
        a[:, :300] = 255
        b = a.copy()
        b[100:140, 100:140] = 0
        value = compare.ssim(a, b)
        assert 0.9 < value <= 1.0, f"SSIM {value} for a nearly identical pair"

    def test_ssim_is_one_for_identical_images(self) -> None:
        a = np.zeros((128, 128), np.uint8)
        a[:, :64] = 255
        assert compare.ssim(a, a.copy()) == pytest.approx(1.0)

    def test_ssim_stays_within_range_on_random_noise(self) -> None:
        rng = np.random.default_rng(5)
        a = rng.integers(0, 256, (200, 200), dtype=np.uint8)
        b = rng.integers(0, 256, (200, 200), dtype=np.uint8)
        assert -1.0 <= compare.ssim(a, b) <= 1.0

    def test_strip_processing_matches_a_single_pass(self) -> None:
        """Strip boundaries must not change the answer."""
        rng = np.random.default_rng(9)
        a = rng.integers(0, 256, (600, 120), dtype=np.uint8)
        b = np.clip(a.astype(np.int16) + rng.integers(-8, 9, a.shape), 0, 255).astype(np.uint8)

        whole = compare.compare_arrays(a, b)
        # 600 rows is more than two strips at the default 256, so this exercises
        # the halo/trim arithmetic rather than a single strip.
        assert whole.ssim == pytest.approx(compare.ssim(a, b), abs=1e-9)
        assert 0.0 <= whole.rmse <= 255.0

    def test_memory_stays_bounded_on_a_large_pair(self) -> None:
        """A 16 MP pair used to need ~2 GB; strip processing keeps it small.

        Measured against a generous ceiling so the test is not flaky, but far
        below the ~8 GB the naive formulation extrapolated to at the 64 MP cap.
        """
        import resource

        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        a = np.zeros((4000, 4000), np.uint8)
        a[:, :2000] = 255
        b = a.copy()
        b[1000:1100, 1000:1100] = 0
        result = compare.compare_arrays(a, b)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        assert result.ssim > 0.99
        growth = after - before
        assert growth < 1200, f"peak RSS grew by {growth:.0f} MB for a 16 MP comparison"

    def test_hash_downscaling_does_not_require_a_full_float_copy(self) -> None:
        """Hashing narrows to uint8 before resizing, not after."""
        a = np.zeros((3000, 3000), np.uint8)
        a[:, :1500] = 255
        hashes = compare.perceptual_hashes(Image.fromarray(a, "L"))
        assert len(hashes["phash"]) == 16


class TestOutputNamesCannotChooseTheirType:
    """A generated image must never be served with a caller-chosen type."""

    def test_the_extension_is_forced_to_match_the_codec(self, config, jail) -> None:
        from pictor_mcp.backends import BackendRegistry
        from pictor_mcp.imaging.fonts import FontIndex
        from pictor_mcp.outputs import ResultBuilder
        from pictor_mcp.security.net import SafeFetcher
        from pictor_mcp.security.ratelimit import ConcurrencyGate
        from pictor_mcp.tools.context import ToolContext

        ctx = ToolContext(
            config=config,
            jail=jail,
            loader=ImageLoader(config, jail, SafeFetcher(config.fetch)),
            builder=ResultBuilder(config, jail),
            fonts=FontIndex(()),
            registry=BackendRegistry(),
            gate=ConcurrencyGate(1),
        )
        spec = resolve_output_format("jpeg")
        assert ctx.output_filename("pwn.html", spec, None) == "pwn.jpg"
        assert ctx.output_filename("../../etc/passwd", spec, None).endswith(".jpg")
        assert "/" not in ctx.output_filename("a/b/c", spec, None)
        assert ctx.output_filename(None, spec, None) == "image.jpg"

    def test_served_content_type_is_allow_listed(self) -> None:
        from pictor_mcp.resources import serve_mime

        assert serve_mime("a/b.webp") == ("image/webp", True)
        assert serve_mime("a/b.png") == ("image/png", True)

        # Anything not a known image is an opaque download, never rendered.
        for name in ("pwn.html", "x.svg", "x.txt", "x.js", "x.pdf", "noextension"):
            content_type, inline = serve_mime(name)
            assert content_type == "application/octet-stream", name
            assert inline is False, name

    def test_svg_is_never_rendered(self) -> None:
        """SVG can carry script; it is not on the servable list for that reason."""
        from pictor_mcp.resources import serve_mime

        assert serve_mime("logo.svg") == ("application/octet-stream", False)


class TestBatchErrorsDoNotLeakPaths:
    async def test_a_batch_failure_hides_the_absolute_path(self, sandbox: Sandbox) -> None:
        """Regression: str(OSError) contains the absolute path it failed on."""
        import os
        import sys

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        from .conftest import SRC

        # Plant a regular file where the server will try to create a directory,
        # which makes the internal failure path stringify an absolute path.
        (sandbox.output / "batch").write_text("in the way")

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "pictor_mcp"],
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(SRC),
                "PICTOR_TRANSPORT": "stdio",
                "PICTOR_INPUT_ROOTS": str(sandbox.input),
                "PICTOR_OUTPUT_ROOT": str(sandbox.output),
                "PICTOR_LOG_LEVEL": "ERROR",
            },
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "image_batch",
                {"paths": ["photo.jpg"], "operations": [{"op": "resize", "width": 32}]},
            )
            payload = result.structured_content or {}
            rendered = str(payload)
            assert str(sandbox.root) not in rendered
            assert "/batch" not in rendered or "batch/" not in rendered
            for entry in payload.get("files", []):
                if entry.get("error"):
                    assert str(sandbox.root) not in entry["error"]["message"]


class TestDeadline:
    def test_an_expired_deadline_raises(self) -> None:
        deadline = Deadline(0.0)
        time.sleep(0.01)
        assert deadline.expired is True
        with pytest.raises(LimitExceededError, match="time budget"):
            deadline.check("test")

    def test_a_fresh_deadline_does_not_raise(self) -> None:
        Deadline(60.0).check("test")

    def test_the_pipeline_checks_the_deadline_between_steps(self, sandbox: Sandbox) -> None:
        from pictor_mcp.imaging.fonts import FontIndex
        from pictor_mcp.imaging.pipeline import ResizeOperation, apply_operations

        expired = Deadline(0.0)
        with pytest.raises(LimitExceededError, match="time budget"):
            apply_operations(
                Image.new("RGB", (64, 64)),
                [ResizeOperation(width=32), ResizeOperation(width=16)],
                config=load_config(base_env(sandbox)),
                fonts=FontIndex(()),
                deadline=expired,
            )

    def test_the_config_value_is_a_budget_not_a_hard_interrupt(self) -> None:
        """Documented honestly: the value is used, but cooperatively."""
        limits = Limits(op_timeout_seconds=5.0)
        assert limits.op_timeout_seconds == 5.0


class TestFrameCountingOrder:
    async def test_per_frame_geometry_is_checked_before_frames_are_counted(self, sandbox: Sandbox) -> None:
        """Regression: APNG decodes frames while seeking to count them.

        Counting happens first in the original code, so each seek could load a
        full frame. The per-frame ceiling must be applied before the walk.
        """
        frames = []
        for index in range(6):
            frame = Image.new("RGB", (110, 110), (index * 30, 0, 0))
            ImageDraw.Draw(frame).rectangle((0, 0, 50, 50 + index), fill=(255, 255, 255))
            frames.append(frame)
        frames[0].save(sandbox.inputs("anim2.png"), save_all=True, append_images=frames[1:], duration=50)

        config = load_config(base_env(sandbox, PICTOR_MAX_PIXELS="10000"))
        from pictor_mcp.security.net import SafeFetcher
        from pictor_mcp.security.paths import PathJail

        jail = PathJail(config.input_roots, config.output_root)
        restricted = ImageLoader(config, jail, SafeFetcher(config.fetch))

        loads = 0
        original = Image.Image.load

        def counting_load(self):  # type: ignore[no-untyped-def]
            nonlocal loads
            loads += 1
            return original(self)

        Image.Image.load = counting_load  # type: ignore[assignment]
        try:
            with pytest.raises(LimitExceededError):
                await restricted.load(ImageSource.parse(path="anim2.png"))
        finally:
            Image.Image.load = original  # type: ignore[assignment]

        # The per-frame check fires before the frame table is walked, so no
        # frame is decoded at all.
        assert loads == 0, f"{loads} frames were decoded before the limit applied"


class TestFetchHostWildcard:
    def test_a_star_pattern_means_any_host(self) -> None:
        """Documented as "any public host"; it used to match nothing."""
        from pictor_mcp.security.net import _host_allowed

        assert _host_allowed("example.com", ("*",)) is True
        assert _host_allowed("anything.example", ("*",)) is True

    def test_an_empty_list_means_any_host(self) -> None:
        from pictor_mcp.security.net import _host_allowed

        assert _host_allowed("example.com", ()) is True

    def test_an_explicit_list_still_restricts(self) -> None:
        from pictor_mcp.security.net import _host_allowed

        assert _host_allowed("cdn.example.com", ("*.example.com",)) is True
        assert _host_allowed("example.com", ("*.example.com",)) is False
        assert _host_allowed("evilexample.com", ("*.example.com",)) is False


class TestOutputReadVerifiesItsDescriptor:
    def test_a_symlinked_parent_cannot_redirect_a_read(self, jail, sandbox: Sandbox) -> None:
        """read_output_bytes now re-checks the opened descriptor, as reads do."""
        jail.write_bytes("real/keep.bin", b"safe")
        outside = sandbox.root / "outside"
        outside.mkdir()
        (outside / "keep.bin").write_bytes(b"secret")

        # Replace the intermediate directory with a symlink pointing outside.
        import shutil

        shutil.rmtree(sandbox.output / "real")
        (sandbox.output / "real").symlink_to(outside, target_is_directory=True)

        from pictor_mcp.errors import PathNotAllowedError

        with pytest.raises(PathNotAllowedError):
            jail.read_output_bytes("real/keep.bin")


class TestEncodeFallbackStillWorks:
    def test_high_quality_jpeg_survives_the_encoder_quirk(self) -> None:
        """The 4:4:4 plus optimize combination fails in some libjpeg builds."""
        import numpy as np

        array = np.random.default_rng(3).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        encoded = encode(Image.fromarray(array, "RGB"), resolve_output_format("jpeg"), EncodeOptions(quality=95))
        assert encoded.data.startswith(b"\xff\xd8")


class TestGpuArchitectureDiagnosis:
    """The GPU backend must explain an architecture mismatch, not just fail.

    A PyTorch build only carries kernels for the compute capabilities it was
    compiled for. When the card is newer than the build - an RTX 50-series
    (sm_120) against a pre-CUDA-12.8 wheel - torch imports, sees the device, and
    fails at the first kernel launch. Torch's own warning lists the capabilities
    it supports but not what to do, and the backend only added "the self-test
    could not run a GPU resize", which names neither the card nor the cause.
    """

    @staticmethod
    def _backend(major: int, minor: int, archs: list[str], detail: str = "RTX 5060 Ti"):
        import types

        from pictor_mcp.backends.torch_cuda import TorchCudaBackend

        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index=0: "NVIDIA GeForce RTX 5060 Ti",
            get_device_capability=lambda index=0: (major, minor),
            get_arch_list=lambda: archs,
            empty_cache=lambda: None,
        )
        torch = types.SimpleNamespace(cuda=cuda, __version__="2.4.1", version=types.SimpleNamespace(cuda="12.4"))
        backend = TorchCudaBackend.__new__(TorchCudaBackend)  # skip __init__, which imports torch
        backend._torch = torch
        backend._device_index = 0
        backend._device = "cuda:0"
        backend._detail = detail
        backend._available = False
        backend._disabled_reason = ""
        return backend

    @pytest.mark.parametrize(
        ("tag", "expected"),
        [
            ("sm_50", (5, 0)),
            ("sm_75", (7, 5)),
            ("sm_86", (8, 6)),
            ("sm_90", (9, 0)),
            ("sm_90a", (9, 0)),
            ("sm_100", (10, 0)),
            ("sm_120", (12, 0)),
            ("sm_120a", (12, 0)),
            ("compute_120", (12, 0)),
            ("not-an-arch", None),
        ],
    )
    def test_arch_tags_parse(self, tag: str, expected: tuple[int, int] | None) -> None:
        from pictor_mcp.backends.torch_cuda import _capability

        assert _capability(tag) == expected

    def test_a_card_newer_than_the_build_advises_a_newer_cuda_line(self) -> None:
        """The reported case: RTX 5060 Ti sm_120 against a cu124-era build."""
        backend = self._backend(12, 0, ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"])
        reason = backend._unsupported_architecture()
        assert reason
        assert "sm_120" in reason
        assert "sm_90" in reason, "the capabilities the build does support"
        assert "predates" in reason
        assert "cu128" in reason, "the remedy must name the index to use"

    def test_a_card_older_than_the_build_advises_an_older_cuda_line(self) -> None:
        """The opposite direction: advising 'newer' here would make it worse."""
        backend = self._backend(7, 5, ["sm_80", "sm_86", "sm_90", "sm_120"])
        reason = backend._unsupported_architecture()
        assert "no longer includes" in reason
        assert "cu126" in reason

    @pytest.mark.parametrize(
        "archs",
        [
            ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
            ["sm_90", "sm_120a"],
            ["sm_90", "compute_120"],
        ],
    )
    def test_a_supported_architecture_reports_nothing(self, archs: list[str]) -> None:
        """Including the arch-specific and PTX-only spellings of coverage."""
        assert self._backend(12, 0, archs)._unsupported_architecture() == ""

    def test_an_unreadable_arch_list_is_not_treated_as_a_mismatch(self) -> None:
        """A driver that cannot answer must not disable a working GPU."""
        assert self._backend(12, 0, [])._unsupported_architecture() == ""

    def test_the_reason_reaches_the_capabilities_report(self) -> None:
        """The operator has to be able to find this without reading logs."""
        backend = self._backend(12, 0, ["sm_80", "sm_90"])
        backend._disabled_reason = backend._unsupported_architecture()
        status = backend.status()
        assert status.available is False
        assert "sm_120" in status.to_public_dict()["disabledReason"]

"""CUDA backend tests, driven by a stub ``torch``.

There is no GPU in CI, and the interesting behaviour here is precisely the
behaviour a working GPU would exercise: deciding whether the device can be used,
and falling back cleanly when it cannot. A stub is therefore the only way to
test it at all - and it caught a defect that a real GPU would merely have
reported as "acceleration: pillow (cpu)":

`_load()` set `_available = True` *after* `_run_self_test()`, while `resize()`
refuses to touch the device unless `available()` is true. The self-test resizes,
so it always found the backend unavailable, recorded "could not run a GPU
resize", and disabled itself. The GPU path was dead code on every machine,
working card or not.

The stub implements the small slice of the torch API the backend uses, with real
array arithmetic, so the self-test's CPU-versus-GPU comparison actually runs.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest
from PIL import Image


class _no_op_context:
    """Stand-in for ``torch.inference_mode()``."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


class _FakeTensor:
    """Minimal NCHW tensor backed by numpy, enough for the backend's pipeline."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.calls: list[str] = []

    # -- conversion. `.to()` is called both as `to(device=..., dtype=...)` and
    # positionally as `to(torch.uint8)`, so it accepts either.
    def to(self, *args: Any, device: Any = None, dtype: Any = None) -> _FakeTensor:
        self.calls.append("to")
        if args:
            dtype = args[0]
        if dtype is not None:
            self.array = self.array.astype(dtype)
        return self

    def cpu(self) -> _FakeTensor:
        self.calls.append("cpu")
        return self

    def numpy(self) -> np.ndarray:
        return self.array

    # -- shape
    def permute(self, *order: int) -> _FakeTensor:
        self.array = self.array.transpose(order)
        return self

    def unsqueeze(self, axis: int) -> _FakeTensor:
        self.array = np.expand_dims(self.array, axis)
        return self

    def squeeze(self, axis: int) -> _FakeTensor:
        self.array = np.squeeze(self.array, axis=axis)
        return self

    # -- value
    def clamp_(self, low: float, high: float) -> _FakeTensor:
        self.array = np.clip(self.array, low, high)
        return self

    def contiguous(self) -> _FakeTensor:
        self.array = np.ascontiguousarray(self.array)
        return self


class _FakeFunctional:
    def __init__(self, recorder: dict[str, int]) -> None:
        self._recorder = recorder

    def interpolate(
        self,
        tensor: _FakeTensor,
        size: tuple[int, int] | None = None,
        mode: str = "nearest",
        antialias: bool = False,
    ) -> _FakeTensor:
        """Bilinear resize over the numpy buffer, so the self-test can compare."""
        self._recorder["interpolate"] += 1
        self._recorder.setdefault("modes", []).append(mode)
        if size is None:  # pragma: no cover - the backend always passes a size
            return tensor

        # (1, C, H, W) -> (H, W, C)
        nchw = tensor.array
        chw = nchw[0]
        hwc = chw.transpose(1, 2, 0)
        # torch's `size` is (H, W); Pillow's resize takes (width, height).
        height, width = size
        single_channel = hwc.shape[2] == 1
        # Pillow has no (H, W, 1) mode, so a grayscale tensor is squeezed for the
        # resize and expanded again afterwards.
        source = hwc[:, :, 0] if single_channel else hwc
        image = Image.fromarray(source.astype(np.uint8))
        resized = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))
        if single_channel:
            resized = resized[:, :, np.newaxis]
        return _FakeTensor(resized[np.newaxis, ...].transpose(0, 3, 1, 2))


class FakeTorch(types.ModuleType):
    """A stub ``torch`` module with configurable device properties."""

    def __init__(
        self,
        *,
        capability: tuple[int, int] = (12, 0),
        arch_list: list[str] | None = None,
        device_name: str = "NVIDIA GeForce RTX 5060 Ti",
        cuda_version: str = "12.8",
        torch_version: str = "2.11.0",
        device_count: int = 1,
        available: bool = True,
        interpolate_raises: Exception | None = None,
    ) -> None:
        super().__init__("torch")
        self.recorder: dict[str, Any] = {"interpolate": 0}
        self.__version__ = torch_version
        self.version = types.SimpleNamespace(cuda=cuda_version)

        backend_self = self

        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                return available

            @staticmethod
            def device_count() -> int:
                return device_count

            @staticmethod
            def get_device_name(index: int = 0) -> str:
                return device_name

            @staticmethod
            def get_device_capability(index: int = 0) -> tuple[int, int]:
                return capability

            @staticmethod
            def get_arch_list() -> list[str]:
                return list(arch_list) if arch_list is not None else ["sm_75", "sm_80", "sm_90", "sm_120"]

            @staticmethod
            def empty_cache() -> None:
                backend_self.recorder["empty_cache"] = backend_self.recorder.get("empty_cache", 0) + 1

        functional = _FakeFunctional(self.recorder)
        if interpolate_raises is not None:

            def _raising(*args: Any, **kwargs: Any) -> Any:
                self.recorder["interpolate"] += 1
                raise interpolate_raises

            functional.interpolate = _raising  # type: ignore[method-assign]

        self.cuda = _Cuda()
        self.nn = types.SimpleNamespace(functional=functional)
        # The backend names these directly: `dtype=torch.float32` and `.to(torch.uint8)`.
        self.float32 = np.float32
        self.uint8 = np.uint8

        # `torch.inference_mode()` wraps the interpolate call. A no-op context
        # manager is enough: the point of the test is the control flow around the
        # device, not autograd.
        self.inference_mode = _no_op_context

    # -- tensor construction
    def from_numpy(self, array: np.ndarray) -> _FakeTensor:
        self.recorder["from_numpy"] = self.recorder.get("from_numpy", 0) + 1
        return _FakeTensor(np.array(array, copy=True))


@pytest.fixture
def stub_torch(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``torch`` and reload the backend module around it."""
    created: list[FakeTorch] = []

    def install(**kwargs: Any) -> FakeTorch:
        module = FakeTorch(**kwargs)
        created.append(module)
        monkeypatch.setitem(sys.modules, "torch", module)
        return module

    return install


def _backend(install, **kwargs: Any):
    from pictor_mcp.backends.torch_cuda import TorchCudaBackend

    install(**kwargs)
    return TorchCudaBackend()


class TestInitialisationOrdering:
    """Regression: the self-test must be able to reach the device."""

    def test_the_self_test_actually_runs_a_gpu_resize(self, stub_torch) -> None:
        """The bug: `resize()` bailed out because availability was set last."""
        torch = stub_torch(arch_list=["sm_75", "sm_80", "sm_90", "sm_120"])
        from pictor_mcp.backends.torch_cuda import TorchCudaBackend

        backend = TorchCudaBackend()
        assert torch.recorder["interpolate"] >= 1, "the self-test never reached the GPU"
        assert backend.available() is True
        assert backend._disabled_reason == ""
        assert "RTX 5060 Ti" in backend.device or backend.device.startswith("cuda")

    def test_a_supported_card_is_reported_as_available(self, stub_torch) -> None:
        backend = _backend(stub_torch)
        assert backend.available() is True
        assert backend.status().available is True

    def test_the_backend_reports_the_device_in_its_status(self, stub_torch) -> None:
        backend = _backend(stub_torch)
        status = backend.status()
        assert status.device == "cuda:0"
        assert "RTX 5060 Ti" in status.detail
        assert "2.11.0" in status.detail


class TestUnusableDevices:
    """Every reason a device cannot be used must be reported, not guessed at."""

    def test_an_architecture_mismatch_is_explained(self, stub_torch) -> None:
        backend = _backend(stub_torch, capability=(12, 0), arch_list=["sm_50", "sm_60", "sm_90"])
        assert backend.available() is False
        reason = backend._disabled_reason
        assert "sm_120" in reason
        assert "cu128" in reason

    def test_a_failing_kernel_names_the_underlying_error(self, stub_torch) -> None:
        """Without the cause, the operator only learns that nothing ran."""
        backend = _backend(
            stub_torch,
            interpolate_raises=RuntimeError("no kernel image is available for execution on the device"),
        )
        assert backend.available() is False
        assert "no kernel image" in backend._disabled_reason

    def test_no_cuda_is_reported_without_a_reason(self, stub_torch) -> None:
        backend = _backend(stub_torch, available=False)
        assert backend.available() is False
        assert backend._disabled_reason == ""
        assert "not available" in backend._detail

    def test_a_missing_device_index_is_reported(self, stub_torch) -> None:
        backend = _backend(stub_torch, device_count=0)
        assert backend.available() is False

    def test_resize_returns_none_when_unavailable(self, stub_torch) -> None:
        """The contract every caller relies on: None means "use the CPU"."""
        backend = _backend(stub_torch, available=False)
        assert backend.resize(Image.new("RGB", (8, 8)), (4, 4), Image.Resampling.BILINEAR) is None


class TestResize:
    """The accelerated path must produce a correct image."""

    def test_resize_matches_pillow_within_tolerance(self, stub_torch) -> None:
        backend = _backend(stub_torch)
        rng = np.random.default_rng(7)
        source = Image.fromarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8), "RGB")

        gpu = backend.resize(source, (37, 53), Image.Resampling.BILINEAR)
        assert gpu is not None
        assert gpu.size == (37, 53)
        assert gpu.mode == "RGB"

        cpu = source.resize((37, 53), Image.Resampling.BILINEAR)
        difference = float(np.mean(np.abs(np.asarray(gpu, np.float32) - np.asarray(cpu, np.float32))))
        assert difference < 2.0, f"stub resize diverged from Pillow by {difference}"

    @pytest.mark.parametrize(
        ("mode", "channels"),
        [("L", 1), ("LA", 2), ("RGB", 3), ("RGBA", 4)],
    )
    def test_every_supported_mode_round_trips(self, stub_torch, mode: str, channels: int) -> None:
        backend = _backend(stub_torch)
        image = Image.new(mode, (16, 16))
        result = backend.resize(image, (8, 8), Image.Resampling.BILINEAR)
        assert result is not None
        assert result.mode == mode
        assert result.size == (8, 8)

    def test_an_unsupported_mode_defers_to_the_cpu(self, stub_torch) -> None:
        """A palette image has no meaningful float representation to interpolate."""
        backend = _backend(stub_torch)
        assert backend.resize(Image.new("P", (16, 16)), (8, 8), Image.Resampling.NEAREST) is None

    def test_lanczos_is_approximated_by_bicubic(self, stub_torch) -> None:
        """Torch has no Lanczos kernel; the substitution is deliberate."""
        backend = _backend(stub_torch)
        backend.resize(Image.new("RGB", (32, 32)), (16, 16), Image.Resampling.LANCZOS)
        assert backend._torch.recorder["modes"][-1] == "bicubic"

    def test_nearest_maps_to_the_nearest_kernel(self, stub_torch) -> None:
        """antialias=True is invalid for nearest in torch, so it must be off."""
        backend = _backend(stub_torch)
        backend.resize(Image.new("RGB", (32, 32)), (16, 16), Image.Resampling.NEAREST)
        assert backend._torch.recorder["modes"][-1] == "nearest"

    def test_a_degenerate_target_is_refused(self, stub_torch) -> None:
        backend = _backend(stub_torch)
        assert backend.resize(Image.new("RGB", (8, 8)), (0, 8), Image.Resampling.BILINEAR) is None

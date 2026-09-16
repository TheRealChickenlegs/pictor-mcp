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


class _FakeTensor:
    """Minimal NCHW tensor backed by numpy, enough for the backend's pipeline.

    Emulates the two torch rules that the real backend got wrong, because a
    permissive stub passes code that fails on hardware:

      * ``torch.from_numpy`` refuses (in practice: warns and reserves undefined
        behaviour) for a non-writable buffer, so the stub records writability.
      * a tensor created inside ``torch.inference_mode()`` is an *inference
        tensor*, and in-place updates to one are forbidden once the mode has
        exited. ``clamp_`` reproduces that error exactly.
    """

    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.calls: list[str] = []
        self.is_inference = False
        self.owner: FakeTorch | None = None

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
        self.calls.append("clamp_")
        owner = self.owner
        if self.is_inference and owner is not None and owner.inference_depth == 0:
            raise RuntimeError(
                "Inplace update to inference tensor outside InferenceMode is not allowed."
                "You can make a clone to get a normal tensor before doing inplace update."
            )
        self.array = np.clip(self.array, low, high)
        return self

    def contiguous(self) -> _FakeTensor:
        self.array = np.ascontiguousarray(self.array)
        return self


class _FakeFunctional:
    def __init__(self, owner: FakeTorch) -> None:
        self._owner = owner
        self._recorder = owner.recorder

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
        out = self._owner._make_tensor(resized[np.newaxis, ...].transpose(0, 3, 1, 2))
        return out


class _InferenceMode:
    """Context manager mirroring ``torch.inference_mode()``."""

    def __init__(self, owner: FakeTorch) -> None:
        self._owner = owner

    def __enter__(self) -> None:
        self._owner._inference_depth += 1

    def __exit__(self, *exc: object) -> bool:
        self._owner._inference_depth -= 1
        return False


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

        functional = _FakeFunctional(self)
        if interpolate_raises is not None:

            def _raising(*args: Any, **kwargs: Any) -> Any:
                self.recorder["interpolate"] += 1
                raise interpolate_raises

            functional.interpolate = _raising  # type: ignore[method-assign]

        self._inference_depth = 0
        self.inference_mode = self._inference_mode

        self.cuda = _Cuda()
        self.nn = types.SimpleNamespace(functional=functional)
        # The backend names these directly: `dtype=torch.float32` and `.to(torch.uint8)`.
        self.float32 = np.float32
        self.uint8 = np.uint8

    # -- inference mode
    @property
    def inference_depth(self) -> int:
        return self._inference_depth

    def _inference_mode(self) -> Any:
        return _InferenceMode(self)

    def _make_tensor(self, array: np.ndarray) -> _FakeTensor:
        tensor = _FakeTensor(np.array(array, copy=True))
        tensor.owner = self
        tensor.is_inference = self._inference_depth > 0
        return tensor

    # -- tensor construction
    def from_numpy(self, array: np.ndarray) -> _FakeTensor:
        self.recorder["from_numpy"] = self.recorder.get("from_numpy", 0) + 1
        # Recorded rather than enforced: torch warns here, and the test asserts
        # on it, so the contract is explicit without the stub being stricter
        # than the library it stands in for.
        self.recorder.setdefault("buffer_writable", []).append(bool(array.flags.writeable))
        return self._make_tensor(array)

    # -- out-of-place clamp, the form that is legal outside inference mode
    def clamp(self, tensor: _FakeTensor, low: float, high: float) -> _FakeTensor:
        self.recorder["clamp"] = self.recorder.get("clamp", 0) + 1
        out = self._make_tensor(np.clip(tensor.array, low, high))
        out.is_inference = tensor.is_inference
        return out


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

    def test_the_buffer_handed_to_torch_is_writable(self, stub_torch) -> None:
        """torch.from_numpy needs a writable buffer.

        A PIL image exposes a read-only one, and `np.ascontiguousarray` returns
        the *same* object when the array is already contiguous - which it is,
        straight from Pillow - so it does not copy and the read-only flag reaches
        torch. Torch warns and reserves the right to undefined behaviour on
        write, which is not a state to leave a resize in.
        """
        backend = _backend(stub_torch)
        backend.resize(Image.new("RGB", (32, 32)), (16, 16), Image.Resampling.BILINEAR)
        writable = backend._torch.recorder.get("buffer_writable", [])
        assert writable, "torch.from_numpy was never called"
        assert all(writable), "a read-only buffer was handed to torch"

    def test_no_inplace_update_happens_outside_inference_mode(self, stub_torch) -> None:
        """Tensors made inside inference_mode reject in-place updates after it.

        The stub raises the same RuntimeError torch does, so this fails if the
        clamp ever moves back outside the `with` block or becomes the in-place
        form again.
        """
        backend = _backend(stub_torch)
        result = backend.resize(Image.new("RGB", (32, 32)), (16, 16), Image.Resampling.BILINEAR)
        assert result is not None, backend._last_error
        recorder = backend._torch.recorder
        assert recorder.get("clamp", 0) >= 1, "the out-of-place clamp was not used"

    def test_a_degenerate_target_is_refused(self, stub_torch) -> None:
        backend = _backend(stub_torch)
        assert backend.resize(Image.new("RGB", (8, 8)), (0, 8), Image.Resampling.BILINEAR) is None

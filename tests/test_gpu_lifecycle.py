"""CUDA remains absent from the server process until a GPU operation needs it."""

from __future__ import annotations

import builtins
import threading

from PIL import Image

from pictor_mcp.backends.isolated_cuda import IsolatedCudaBackend


def test_registry_construction_and_capability_report_do_not_import_torch(config, monkeypatch) -> None:
    """A GPU-enabled config must not initialize CUDA during server startup/status."""
    from pictor_mcp.server import build_context

    object_import = builtins.__import__
    imports: list[str] = []

    def import_without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            imports.append(name)
            raise AssertionError("torch import attempted in the MCP server process")
        return object_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    context = build_context(config)
    registry = context.registry
    public = registry.to_public_dict()

    assert imports == []
    assert registry.gpu_device_info() == []
    assert public["backends"][0]["available"] is False
    assert "no VRAM" in public["backends"][0]["detail"]
    assert public["gpuWorkerIdleSeconds"] == config.gpu_idle_seconds
    assert public["gpuWorkerIsolation"].startswith("separate process")
    assert not registry.backends[0].initialized


def test_gpu_idle_timer_waits_for_tool_calls_and_then_stops_worker(monkeypatch) -> None:
    backend = IsolatedCudaBackend(idle_seconds=0.03)
    stopped = threading.Event()
    backend._process = object()  # Timer only checks whether a worker exists.
    monkeypatch.setattr(backend, "_terminate_locked", lambda: stopped.set())

    backend.begin_tool_call()
    backend.end_tool_call()
    assert not stopped.wait(0.01)

    backend.begin_tool_call()  # An intervening CPU tool call refreshes the lease.
    assert not stopped.wait(0.04)
    backend.end_tool_call()
    assert stopped.wait(0.5)


def test_zero_idle_timeout_releases_worker_at_end_of_call(monkeypatch) -> None:
    backend = IsolatedCudaBackend(idle_seconds=0)
    stopped = threading.Event()
    backend._process = object()
    monkeypatch.setattr(backend, "_terminate_locked", lambda: stopped.set())

    backend.begin_tool_call()
    backend.end_tool_call()
    assert stopped.is_set()


def test_gpu_enabled_status_is_lazy_before_first_resize(config) -> None:
    from pictor_mcp.backends import build_registry

    registry = build_registry(config)
    backend = registry.backends[0]

    assert backend.available() is True
    assert backend.status().available is False
    assert "no VRAM allocated" in backend.status().detail
    assert registry.resampler_for(Image.new("RGB", (2000, 2000))) is backend
    assert backend.initialized is False  # Selecting a proxy does not spawn the child.
    registry.gpu_worker.close()

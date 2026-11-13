"""CUDA worker process entry point.

This module deliberately imports neither torch nor ONNX Runtime at module scope.
The child is spawned only on the first eligible GPU operation; terminating it
releases its CUDA context, something ``torch.cuda.empty_cache`` cannot promise.
"""

from __future__ import annotations

import multiprocessing.connection
import os
from contextlib import suppress


def run(connection: multiprocessing.connection.Connection, resize_enabled: bool = True) -> None:
    """Serve resize and ML-background requests until the parent closes the pipe."""
    try:
        backend = None
        device_info: list[dict[str, object]] = []
        if resize_enabled:
            from .torch_cuda import TorchCudaBackend

            backend = TorchCudaBackend()
            status = backend.status()
        else:
            status = None
        if status is not None and status.available and backend is not None and backend._torch is not None:
            torch = backend._torch
            with suppress(Exception):
                properties = torch.cuda.get_device_properties(backend._device_index)
                free, total = torch.cuda.mem_get_info(backend._device_index)
                device_info.append(
                    {
                        "index": backend._device_index,
                        "name": properties.name,
                        "computeCapability": f"{properties.major}.{properties.minor}",
                        "totalMemoryBytes": total,
                        "freeMemoryBytes": free,
                    }
                )
        status_payload = (
            status.to_public_dict()
            if status is not None
            else {
                "name": "torch-cuda",
                "kind": "resize",
                "available": False,
                "device": "cpu",
                "detail": "CUDA resizing disabled by PICTOR_GPU=off",
            }
        )
        connection.send(("ready", status_payload, device_info))

        parent_pid = os.getppid()
        while True:
            try:
                if not connection.poll(1.0):
                    # If the MCP process is killed abruptly, do not orphan a GPU
                    # worker that would otherwise keep the CUDA context alive.
                    if os.getppid() != parent_pid:
                        break
                    continue
                request = connection.recv()
            except (EOFError, OSError):
                break
            if not isinstance(request, tuple) or not request:
                connection.send(("error", "invalid worker request"))
                continue

            operation = request[0]
            try:
                if operation == "resize":
                    _, mode, size, raw, target, pil_filter = request
                    from PIL import Image

                    image = Image.frombytes(mode, size, raw)
                    result = backend.resize(image, target, pil_filter) if backend is not None else None
                    if result is None:
                        connection.send(("fallback",))
                    else:
                        connection.send(("image", result.mode, result.size, result.tobytes()))
                elif operation == "remove_background":
                    _, mode, size, raw, model, background = request
                    from PIL import Image

                    from ..imaging.background import remove_background_ml

                    image = Image.frombytes(mode, size, raw)
                    result, notes = remove_background_ml(image, model=model, background=background)
                    connection.send(("image", result.mode, result.size, result.tobytes(), notes))
                elif operation == "shutdown":
                    break
                else:
                    connection.send(("error", "unsupported worker operation"))
            except BaseException as exc:
                # Keep library diagnostics out of the IPC response; the caller
                # gets a neutral fallback, and server-side logs contain the type.
                connection.send(("error", type(exc).__name__))
    except BaseException as exc:
        with suppress(Exception):
            connection.send(("startup_error", type(exc).__name__))
    finally:
        connection.close()

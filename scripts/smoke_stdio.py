#!/usr/bin/env python
"""Minimal end-to-end connectivity check.

Spawns the server over stdio, completes the MCP handshake, and asserts the
expected tools are advertised. Intended as a post-install or post-deploy smoke
test - it validates that the package is importable, the entry point works, the
protocol handshake completes, and tool registration succeeded, without needing
an image fixture.

Usage::

    python scripts/smoke_stdio.py [--input DIR] [--output DIR]

Exits non-zero with a readable message on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anyio

#: Tools that must always be present. Deliberately a subset: adding a tool
#: should not break this check, but losing one should.
REQUIRED_TOOLS = frozenset(
    {
        "image_capabilities",
        "image_info",
        "image_convert",
        "image_resize",
        "image_transform",
        "image_batch",
        "image_compare",
    }
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Input root (defaults to a temp dir).")
    parser.add_argument("--output", default=None, help="Output root (defaults to a temp dir).")
    parser.add_argument(
        "--expect-tools",
        type=int,
        default=0,
        help="Also assert the total tool count equals this value (0 to skip).",
    )
    args = parser.parse_args()

    import tempfile

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    temporary = None
    if args.input and args.output:
        input_root, output_root = Path(args.input), Path(args.output)
        input_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory()
        input_root = Path(temporary.name) / "input"
        output_root = Path(temporary.name) / "output"
        input_root.mkdir()
        output_root.mkdir()

    async def run() -> int:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "pictor_mcp"],
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "PICTOR_TRANSPORT": "stdio",
                "PICTOR_INPUT_ROOTS": str(input_root),
                "PICTOR_OUTPUT_ROOT": str(output_root),
                "PICTOR_LOG_LEVEL": "ERROR",
            },
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"connected: {init.server_info.name} {init.server_info.version}")
            print(f"protocol:  {session.protocol_version}")

            tools = await session.list_tools()
            names = sorted(tool.name for tool in tools.tools)
            print(f"tools:     {len(names)}")

            missing = REQUIRED_TOOLS - set(names)
            if missing:
                print(f"FAIL: missing tools: {sorted(missing)}", file=sys.stderr)
                return 1
            if args.expect_tools and len(names) != args.expect_tools:
                print(
                    f"FAIL: expected {args.expect_tools} tools, got {len(names)}",
                    file=sys.stderr,
                )
                return 1

            # Exercise one real call so a broken tool registration or result
            # envelope is caught here rather than by a user.
            result = await session.call_tool("image_capabilities", {})
            if result.is_error:
                print("FAIL: image_capabilities returned an error", file=sys.stderr)
                return 1
            payload = result.structured_content or {}
            if not payload.get("ok"):
                print("FAIL: image_capabilities payload was not ok", file=sys.stderr)
                return 1
            print(f"accel:     {payload['backends']['active']}")
            print(f"formats:   {len(payload['formats']['writable'])} writable")
        return 0

    try:
        return anyio.run(run)
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

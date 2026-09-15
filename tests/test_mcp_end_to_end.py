"""End-to-end tests against a real MCP server.

These spawn the actual server over stdio and drive it with the real MCP client,
so they exercise argument-schema generation, the tool guard, the result
envelope and the protocol handshake exactly as a client sees them. Unit tests
cannot catch a schema that fails to serialize or an error message the SDK
swallows.
"""

from __future__ import annotations

import json
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .conftest import Sandbox

pytestmark = pytest.mark.anyio


@pytest.fixture
async def session(server_env):
    """A live client session against a freshly spawned server process."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "pictor_mcp"],
        env=server_env,
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        yield client


def _text(result) -> str:
    return "\n".join(block.text for block in result.content if block.type == "text")


def _structured(result) -> dict:
    return result.structured_content or {}


class TestProtocolSurface:
    async def test_initializes_and_lists_every_tool(self, session: ClientSession) -> None:
        tools = await session.list_tools()
        names = {tool.name for tool in tools.tools}
        assert names == {
            "image_capabilities",
            "image_info",
            "image_convert",
            "image_resize",
            "image_compress",
            "image_crop",
            "image_rotate",
            "image_thumbnail",
            "image_transform",
            "image_watermark",
            "image_background_remove",
            "image_batch",
            "image_compare",
            "image_optimize_web",
        }

    async def test_every_tool_has_a_usable_description(self, session: ClientSession) -> None:
        tools = await session.list_tools()
        for tool in tools.tools:
            assert tool.description and len(tool.description) > 30, tool.name
            assert tool.input_schema

    async def test_tool_order_is_deterministic(self, session: ClientSession) -> None:
        """A stable order is a spec SHOULD, and it helps client-side caching."""
        first = [tool.name for tool in (await session.list_tools()).tools]
        second = [tool.name for tool in (await session.list_tools()).tools]
        assert first == second

    async def test_tools_advertise_an_output_schema(self, session: ClientSession) -> None:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        schema = tools["image_resize"].output_schema or {}
        assert "outputs" in schema.get("properties", {})
        assert "sizeChange" in schema.get("properties", {})

    async def test_transform_exposes_the_operation_union(self, session: ClientSession) -> None:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        operations = tools["image_transform"].input_schema["properties"]["operations"]
        assert operations["type"] == "array"
        rendered = json.dumps(operations)
        for name in ("resize", "crop", "rotate", "watermark_text", "background_remove"):
            assert name in rendered

    async def test_resource_template_is_published(self, session: ClientSession) -> None:
        templates = await session.list_resource_templates()
        assert any("pictor://outputs" in template.uri_template for template in templates.resource_templates)


class TestResultEnvelope:
    async def test_result_carries_text_image_and_link(self, session: ClientSession) -> None:
        result = await session.call_tool(
            "image_resize",
            {"path": "photo.jpg", "width": 80, "fit": "contain", "return_image": True},
        )
        assert result.is_error is False
        types = [block.type for block in result.content]
        assert types[0] == "text"
        assert "image" in types
        assert "resource_link" in types

    async def test_text_block_is_self_contained(self, session: ClientSession) -> None:
        """A client that renders only text must still learn everything useful."""
        result = await session.call_tool("image_resize", {"path": "photo.jpg", "width": 80})
        text = _text(result)
        assert "Output:" in text
        assert "80x" in text
        assert "resized/" in text
        assert "Size:" in text

    async def test_structured_content_is_complete(self, session: ClientSession) -> None:
        result = await session.call_tool("image_resize", {"path": "photo.jpg", "width": 80})
        payload = _structured(result)
        assert payload["ok"] is True
        assert payload["operation"] == "image_resize"
        output = payload["outputs"][0]
        assert output["width"] == 80
        assert output["mimeType"] == "image/jpeg"
        assert len(output["sha256"]) == 64
        assert output["byteSize"] > 0

    async def test_size_change_is_reported(self, session: ClientSession) -> None:
        result = await session.call_tool(
            "image_compress", {"path": "photo.jpg", "quality": 40, "target_format": "jpeg"}
        )
        change = _structured(result)["sizeChange"]
        assert change["inputBytes"] > 0
        assert change["outputBytes"] > 0
        assert "savedPercent" in change

    async def test_inline_can_be_disabled(self, session: ClientSession) -> None:
        result = await session.call_tool("image_resize", {"path": "photo.jpg", "width": 64, "return_image": False})
        assert all(block.type != "image" for block in result.content)
        assert _structured(result)["inlineImageIncluded"] is False

    async def test_base64_can_be_requested(self, session: ClientSession) -> None:
        result = await session.call_tool(
            "image_resize",
            {"path": "photo.jpg", "width": 48, "return_image": False, "return_base64": True},
        )
        output = _structured(result)["outputs"][0]
        assert output["base64"]

    async def test_output_file_exists_on_disk(self, session: ClientSession, sandbox: Sandbox) -> None:
        result = await session.call_tool("image_resize", {"path": "photo.jpg", "width": 40})
        relative = _structured(result)["outputs"][0]["path"]
        assert (sandbox.output / relative).is_file()


class TestEveryToolRuns:
    """One smoke call per tool, so a broken signature cannot hide."""

    @pytest.mark.parametrize(
        ("tool", "arguments"),
        [
            ("image_capabilities", {}),
            ("image_info", {"path": "photo.jpg"}),
            ("image_convert", {"path": "photo.jpg", "target_format": "png"}),
            ("image_resize", {"path": "photo.jpg", "width": 64}),
            ("image_compress", {"path": "photo.jpg", "quality": 50}),
            ("image_crop", {"path": "photo.jpg", "aspect_ratio": 1.0}),
            ("image_rotate", {"path": "photo.jpg", "angle": 90}),
            ("image_thumbnail", {"path": "photo.jpg", "size": [48, 48]}),
            (
                "image_transform",
                {
                    "path": "photo.jpg",
                    "operations": [
                        {"op": "resize", "width": 96, "fit": "cover", "height": 96},
                        {"op": "sharpen", "amount": 1.0},
                    ],
                },
            ),
            ("image_watermark", {"path": "photo.jpg", "text": "TEST", "font_size": 20}),
            ("image_watermark", {"path": "photo.jpg", "watermark_path": "logo.png"}),
            (
                "image_background_remove",
                {"path": "flat.png", "method": "color", "tolerance": 30},
            ),
            ("image_batch", {"paths": ["photo.jpg"], "operations": [{"op": "resize", "width": 32}]}),
            ("image_compare", {"path": "photo.jpg", "compare_to_path": "photo.png"}),
            ("image_optimize_web", {"path": "photo.jpg", "widths": [64, 96]}),
        ],
    )
    async def test_tool_succeeds(self, session: ClientSession, tool: str, arguments: dict) -> None:
        result = await session.call_tool(tool, arguments)
        assert result.is_error is False, f"{tool} failed: {_text(result)}"
        assert result.content


class TestErrorsReachTheClient:
    """A rejection must be readable, not a generic crash message."""

    @pytest.mark.parametrize(
        ("tool", "arguments", "needle"),
        [
            ("image_info", {"path": "../../../etc/passwd"}, "outside the configured input roots"),
            ("image_info", {"path": "/etc/passwd"}, "outside the configured input roots"),
            ("image_info", {"path": "nested/../../../../etc/passwd"}, "outside"),
            ("image_info", {}, "exactly one"),
            ("image_info", {"path": "photo.jpg", "base64_data": "AAAA"}, "only one"),
            ("image_info", {"path": "missing.png"}, "not found"),
            ("image_info", {"path": "notimage.txt"}, "not a recognisable image"),
            ("image_info", {"path": "broken.jpg"}, "header could not be parsed"),
            ("image_convert", {"path": "photo.jpg", "target_format": "pdf"}, "cannot write"),
            ("image_convert", {"path": "photo.jpg", "target_format": "heic"}, "cannot write"),
            ("image_resize", {"path": "photo.jpg"}, "width, height or percent"),
            ("image_crop", {"path": "photo.jpg"}, "box, aspect_ratio or trim"),
            ("image_watermark", {"path": "photo.jpg"}, "exactly one of"),
            (
                "image_background_remove",
                {"path": "photo.jpg", "output_format": "jpeg", "method": "color"},
                "cannot store transparency",
            ),
            ("image_info", {"path": "photo.jpg", "url": "http://example.com/x.png"}, "only one"),
            (
                "image_transform",
                {"path": "photo.jpg", "operations": [{"op": "resize", "wdith": 5}]},
                "",
            ),
        ],
    )
    async def test_error_is_actionable(self, session: ClientSession, tool: str, arguments: dict, needle: str) -> None:
        result = await session.call_tool(tool, arguments)
        assert result.is_error is True, f"{tool} unexpectedly succeeded"
        if needle:
            assert needle in _text(result), f"got: {_text(result)!r}"

    async def test_error_carries_a_machine_readable_code(self, session: ClientSession) -> None:
        result = await session.call_tool("image_info", {"path": "/etc/passwd"})
        error = _structured(result).get("error")
        assert error is not None, _text(result)
        assert error["code"] == "path_not_allowed"
        assert error["message"]

    async def test_errors_do_not_leak_host_paths(self, session: ClientSession, sandbox: Sandbox) -> None:
        for arguments in ({"path": "/etc/passwd"}, {"path": "../../../etc/shadow"}):
            result = await session.call_tool("image_info", arguments)
            text = _text(result)
            assert "/etc/" not in text
            assert str(sandbox.root) not in text

    async def test_url_fetch_is_disabled_by_default(self, session: ClientSession) -> None:
        result = await session.call_tool("image_info", {"url": "http://example.com/a.png"})
        assert result.is_error is True
        assert "disabled" in _text(result)

    async def test_unknown_tool_is_an_error(self, session: ClientSession) -> None:
        result = await session.call_tool("image_teleport", {})
        assert result.is_error is True

    async def test_invalid_enum_is_rejected_by_the_schema(self, session: ClientSession) -> None:
        result = await session.call_tool(
            "image_transform",
            {"path": "photo.jpg", "operations": [{"op": "resize", "fit": "squishy", "width": 10}]},
        )
        assert result.is_error is True


class TestFormatsAndBehaviour:
    async def test_capabilities_reports_the_security_posture(self, session: ClientSession) -> None:
        result = await session.call_tool("image_capabilities", {})
        payload = _structured(result)
        security = payload["security"]
        assert security["networkFetch"]["enabled"] is False
        assert security["metadataStrippedByDefault"] is True
        assert "169.254" not in json.dumps(security)  # the block list is described, not enumerated by IP
        assert payload["backends"]["active"].startswith("pillow")
        assert payload["protocol"]["structuredContent"] is True

    async def test_capabilities_lists_only_writable_formats(self, session: ClientSession) -> None:
        payload = _structured(await session.call_tool("image_capabilities", {}))
        writable = {item["format"] for item in payload["formats"]["writable"]}
        assert {"jpeg", "png", "webp"} <= writable
        assert "pdf" not in writable
        assert "eps" not in writable

    async def test_animation_survives_conversion(self, session: ClientSession, sandbox: Sandbox) -> None:
        # Guard the fixture itself: a single-frame GIF would make this vacuous.
        info = await session.call_tool("image_info", {"path": "anim.gif"})
        assert _structured(info)["image"]["frames"] == 4

        result = await session.call_tool("image_convert", {"path": "anim.gif", "target_format": "webp", "quality": 70})
        assert result.is_error is False, _text(result)
        output = _structured(result)["outputs"][0]
        from PIL import Image

        with Image.open(sandbox.output / output["path"]) as reopened:
            assert getattr(reopened, "n_frames", 1) == 4

    async def test_metadata_is_stripped_by_default(self, session: ClientSession, sandbox: Sandbox) -> None:
        result = await session.call_tool("image_convert", {"path": "photo.jpg", "target_format": "jpeg", "quality": 80})
        output = _structured(result)["outputs"][0]
        from PIL import Image

        with Image.open(sandbox.output / output["path"]) as reopened:
            assert not reopened.info.get("exif")

    async def test_batch_reports_per_file_results(self, session: ClientSession) -> None:
        result = await session.call_tool(
            "image_batch",
            {
                "paths": ["photo.jpg", "nested/deep.png"],
                "operations": [{"op": "resize", "width": 32}],
            },
        )
        payload = _structured(result)
        assert payload["processed"] == 2
        assert payload["succeeded"] == 2
        assert all(entry["status"] == "ok" for entry in payload["files"])

    async def test_batch_isolates_a_bad_file(self, session: ClientSession) -> None:
        """One unreadable input must not abort the whole run."""
        result = await session.call_tool(
            "image_batch",
            {
                "paths": ["photo.jpg", "notimage.txt", "photo.png"],
                "operations": [{"op": "resize", "width": 32}],
            },
        )
        payload = _structured(result)
        assert payload["processed"] == 3
        assert payload["succeeded"] == 2
        statuses = {entry["source"]: entry["status"] for entry in payload["files"]}
        assert statuses["notimage.txt"] == "skipped"

    async def test_compare_verifies_an_edit(self, session: ClientSession) -> None:
        """The agent-facing loop: transform, then confirm what changed."""
        resized = await session.call_tool(
            "image_resize", {"path": "photo.jpg", "width": 100, "fit": "fill", "height": 100}
        )
        produced = _structured(resized)["outputs"][0]["path"]
        comparison = await session.call_tool(
            "image_compare",
            {"path": "photo.jpg", "compare_to_path": produced, "align": True},
        )
        metrics = _structured(comparison)["metrics"]
        assert "ssim" in metrics
        assert comparison.is_error is False

    async def test_transform_note_explains_a_read_only_input_format(self, session: ClientSession) -> None:
        result = await session.call_tool("image_info", {"path": "photo.jpg"})
        assert _structured(result)["image"]["format"] == "jpeg"

    async def test_resource_read_returns_the_bytes(self, session: ClientSession) -> None:
        resized = await session.call_tool("image_resize", {"path": "photo.jpg", "width": 48})
        path = _structured(resized)["outputs"][0]["path"]
        resource = await session.read_resource(f"pictor://outputs/{path}")
        assert resource.contents
        assert getattr(resource.contents[0], "blob", None)

    async def test_resource_read_refuses_traversal(self, session: ClientSession) -> None:
        with pytest.raises(Exception):
            await session.read_resource("pictor://outputs/../../etc/passwd")

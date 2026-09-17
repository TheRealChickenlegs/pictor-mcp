"""How a result reaches a chat UI.

Open WebUI does not display the image content block an MCP tool returns
(open-webui discussion #14732): it renders markdown from the assistant's reply.
A result that carries a URL and nothing else therefore produces no picture - the
model summarises the run and the link is lost - so the text has to end with the
exact line to send. These tests pin that, and pin the difference between serving
outputs and not.
"""

from __future__ import annotations

import re

import pytest

from pictor_mcp.config import load_config
from pictor_mcp.models import FileOutput, ImageResult
from pictor_mcp.outputs import ResultBuilder
from pictor_mcp.security.paths import PathJail

from .conftest import Sandbox, base_env

TOKEN = "a-sufficiently-long-token-value"


def _builder(sandbox: Sandbox, **overrides: str) -> ResultBuilder:
    config = load_config(base_env(sandbox, PICTOR_AUTH_TOKEN=TOKEN, **overrides))
    jail = PathJail(config.input_roots, config.output_root)
    jail.ensure_output_root()
    return ResultBuilder(config, jail)


def _result(*, url: str | None, outputs: int = 1) -> ImageResult:
    return ImageResult(
        ok=True,
        operation="image_resize",
        input={"width": 4000, "height": 3000, "format": "jpeg", "byteSize": 3_200_000},
        outputs=[
            FileOutput(
                name=f"photo-w{1200 + index}.webp",
                path=f"resized/photo-w{1200 + index}.webp",
                mime_type="image/webp",
                format="webp",
                byte_size=142_300,
                width=1200 + index,
                height=900,
                sha256="a" * 64,
                url=url and f"{url}?e=1&s=2",
            )
            for index in range(outputs)
        ],
        size_change={"inputBytes": 3_200_000, "outputBytes": 142_300, "savedBytes": 3_057_700, "savedPercent": 95.5},
    )


class TestChatUiDisplay:
    def test_the_summary_ends_with_the_markdown_to_send(self, sandbox: Sandbox) -> None:
        text = _builder(
            sandbox, PICTOR_SERVE_OUTPUTS="true", PICTOR_PUBLIC_BASE_URL="https://pictor.example.com"
        ).summarise(_result(url="https://pictor.example.com/files/resized/photo-w1200.webp"))

        assert text.rstrip().endswith(
            "![photo-w1200.webp](https://pictor.example.com/files/resized/photo-w1200.webp?e=1&s=2)"
        )
        assert "copy this into your reply exactly as written" in text

    def test_every_output_gets_its_own_line(self, sandbox: Sandbox) -> None:
        """A width ladder is several images; the model must not be given one and
        left to guess the rest."""
        text = _builder(
            sandbox, PICTOR_SERVE_OUTPUTS="true", PICTOR_PUBLIC_BASE_URL="https://pictor.example.com"
        ).summarise(_result(url="https://pictor.example.com/files/resized/x.webp", outputs=3))
        markdown = re.findall(r"!\[[^\]]+\]\([^)]+\)", text)
        assert len(markdown) == 3, markdown

    def test_no_markdown_without_a_url(self, sandbox: Sandbox) -> None:
        """Without serving there is nothing to render, and inventing a relative
        link would be worse than saying nothing."""
        text = _builder(sandbox).summarise(_result(url=None))
        assert "![" not in text
        assert "copy this into your reply" not in text

    def test_the_url_is_built_from_the_public_base(self, sandbox: Sandbox) -> None:
        builder = _builder(sandbox, PICTOR_SERVE_OUTPUTS="true", PICTOR_PUBLIC_BASE_URL="https://pictor.example.com")
        assert builder._public_url("resized/photo.webp").startswith(
            "https://pictor.example.com/files/resized/photo.webp?"
        )

    def test_a_link_is_signed_so_the_browser_needs_no_token(self, sandbox: Sandbox) -> None:
        """An <img> tag cannot send a bearer token, so the signature is the
        credential. It must be present, and it must expire."""
        url = _builder(
            sandbox, PICTOR_SERVE_OUTPUTS="true", PICTOR_PUBLIC_BASE_URL="https://pictor.example.com"
        )._public_url("resized/photo.webp")
        assert re.search(r"[?&]e=\d+", url), url
        assert re.search(r"[?&]s=[0-9a-f]{16,}", url), url


class TestOpenWebUiContentShape:
    """What Open WebUI does with the content blocks we return.

    Its `process_tool_result` walks the MCP content list and:

    * keeps ``type: "text"`` items, then sets the tool result to that string
      **only if there is exactly one** - two text blocks become a list, which is
      then ``json.dumps``-ed into a blob in which the markdown is escaped and
      renders as nothing but characters;
    * tries ``get_file_url_from_base64`` for ``type: "image"`` using
      ``item["mimeType"]``, which its own ``model_dump()`` has already renamed to
      ``mime_type``, so the data URI becomes ``data:None;base64,...`` and the
      image is dropped. No MCP server using the official SDK models can avoid
      that, which is why the markdown URL is the only path that displays;
    * ignores anything else, including our ``resource_link`` entries.
    """

    def _content(self, sandbox: Sandbox, **overrides: str):
        result = _result(url="https://pictor.example.com/files/resized/photo-w1200.webp")
        return (
            _builder(sandbox, **overrides)
            .build_model_result(result, inline_bytes=b"\x89PNG fake", inline_mime="image/webp")
            .content
        )

    def test_exactly_one_text_block(self, sandbox: Sandbox) -> None:
        """Two would be JSON-ified by Open WebUI and the markdown would stop
        rendering - a silent failure with no error anywhere."""
        blocks = [block.model_dump(mode="json") for block in self._content(sandbox)]
        assert len([block for block in blocks if block["type"] == "text"]) == 1, blocks

    def test_the_text_block_carries_the_markdown(self, sandbox: Sandbox) -> None:
        blocks = [block.model_dump(mode="json") for block in self._content(sandbox)]
        text = next(block["text"] for block in blocks if block["type"] == "text")
        assert re.search(r"!\[[^\]]+\]\(https://pictor\.example\.com/files/[^)]+\)", text), text

    def test_the_image_block_uses_the_spec_field_name_on_the_wire(self, sandbox: Sandbox) -> None:
        """`mimeType`, not `mime_type`: that is the MCP field name, and the SDK
        model aliases it. Pinned because renaming it to `mime_type` to match
        Open WebUI's bug would break every other client."""
        blocks = [block.model_dump(mode="json", by_alias=True) for block in self._content(sandbox)]
        image = next(block for block in blocks if block["type"] == "image")
        assert image["mimeType"] == "image/webp", image
        assert "mime_type" not in image, image

    @pytest.mark.parametrize(("inline_images", "expected"), [("true", True), ("false", False)])
    def test_an_inline_image_can_be_switched_off(self, sandbox: Sandbox, inline_images: str, expected: bool) -> None:
        """Open WebUI discards the block, so sending it only costs context and
        bandwidth. This exercises the real build path, where the setting is read,
        rather than `build_model_result`, which is handed bytes explicitly."""
        from pictor_mcp.models import FileOutput
        from pictor_mcp.outputs import StoredOutput

        output = StoredOutput(
            file=FileOutput(
                name="photo-w1200.webp",
                path="resized/photo-w1200.webp",
                mime_type="image/webp",
                format="webp",
                byte_size=142_300,
                width=1200,
                height=900,
                sha256="a" * 64,
                url="https://pictor.example.com/files/resized/photo-w1200.webp?e=1&s=2",
            ),
            data=b"\x89PNG fake",
            absolute_path=sandbox.outputs("resized/photo-w1200.webp"),
        )
        builder = _builder(sandbox, PICTOR_INLINE_IMAGES=inline_images)
        result = builder.build_image_result(operation="image_resize", outputs=[output], inline=True)
        kinds = [block.model_dump(mode="json")["type"] for block in result.content]
        assert ("image" in kinds) is expected, kinds


class TestServingIsRequired:
    @pytest.mark.parametrize("key", ["PICTOR_PUBLIC_BASE_URL"])
    def test_a_base_url_alone_changes_nothing_in_the_text(self, sandbox: Sandbox, key: str) -> None:
        """The pairing that matters: a base URL without serving is inert, which
        is why the server warns about it rather than leaving it unexplained."""
        text = _builder(sandbox, **{key: "https://pictor.example.com"}).summarise(_result(url=None))
        assert "https://pictor.example.com" not in text

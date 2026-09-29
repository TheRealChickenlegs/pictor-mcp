"""SVG input: recognition, limits, and isolation from a crashing renderer.

The renderer is a native library that aborts the process on some small documents,
so these tests are doing double duty. They check the feature works, and they check
that the hostile inputs which would otherwise take the whole server down instead
produce an ordinary refused tool call. Every "survives" assertion below is written
so that a regression shows up as a failure rather than as a dead test runner.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from pictor_mcp.config import Config, SvgPolicy, load_config
from pictor_mcp.errors import LimitExceededError, PictorError, UnsupportedFormatError
from pictor_mcp.imaging import vector
from pictor_mcp.imaging.formats import (
    INPUT_FORMATS,
    OUTPUT_FORMATS,
    public_format_catalogue,
    resolve_output_format,
)
from pictor_mcp.imaging.loader import ImageLoader, ImageSource
from pictor_mcp.imaging.vector import (
    NotSvgDocumentError,
    inspect_svg,
    looks_like_svg,
    render_target,
    renderer_available,
)
from pictor_mcp.outputs import default_filename
from pictor_mcp.security.net import SafeFetcher
from pictor_mcp.security.paths import PathJail

from .conftest import Sandbox

pytestmark = pytest.mark.anyio

NS = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"

#: Illustrator's exporter emits exactly this shape: a DOCTYPE with an internal
#: subset declaring namespace entities that the body never references. Rejecting
#: entity declarations outright - a tempting hardening step - would refuse a large
#: share of the SVG files people actually have.
ILLUSTRATOR_SVG = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    "<!-- Generator: Adobe Illustrator 27.0.0, SVG Export Plug-In -->\n"
    '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
    '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [\n'
    '\t<!ENTITY ns_extend "http://ns.adobe.com/Extensibility/1.0/">\n'
    '\t<!ENTITY ns_ai "http://ns.adobe.com/AdobeIllustrator/10.0/">\n'
    "]>\n"
    '<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="120px" height="60px" '
    'viewBox="0 0 120 60" xml:space="preserve">\n'
    '<rect width="120" height="60" fill="#3366cc"/>\n'
    '<circle cx="30" cy="30" r="20" fill="#ffcc00"/>\n'
    "</svg>\n"
)


def svg_document(body: str, *, attrs: str = 'width="16" height="16"') -> bytes:
    return f'<svg xmlns="{NS}" {attrs}>{body}</svg>'.encode()


def pattern_chain(count: int) -> bytes:
    """A shallow document that overflows the renderer's stack.

    Every element is a sibling at XML depth 5, so this is invisible to any nesting
    limit: 300 of them (about 20 KB) sigsegv resvg 0.5.0 deterministically, while
    200 render normally. This is the input that makes in-process rendering
    untenable, and the reason rasterisation happens in a child process.
    """
    body = '<defs><pattern id="p0" width="16" height="16"><rect width="16" height="16"/></pattern>'
    body += "".join(
        f'<pattern id="p{i}" width="16" height="16"><rect width="16" height="16" fill="url(#p{i - 1})"/></pattern>'
        for i in range(1, count)
    )
    return svg_document(body + f'</defs><rect width="16" height="16" fill="url(#p{count - 1})"/>')


def deep_document(depth: int) -> bytes:
    return svg_document("<g>" * depth + '<rect width="16" height="16"/>' + "</g>" * depth)


class TestRecognition:
    """``looks_like_svg`` is a bounded prefix hint, and must stay cheap and fuzzy."""

    @pytest.mark.parametrize(
        "document",
        [
            b'<svg xmlns="http://www.w3.org/2000/svg"/>',
            b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"/>',
            b'  \n\t<svg xmlns="http://www.w3.org/2000/svg"/>',
            b'\xef\xbb\xbf<svg xmlns="http://www.w3.org/2000/svg"/>',
            b'<SVG xmlns="http://www.w3.org/2000/svg"/>',
            b'<svg:svg xmlns:svg="http://www.w3.org/2000/svg"/>',
            b'<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/x.dtd"><svg/>',
        ],
    )
    def test_recognises_the_shapes_real_files_come_in(self, document: bytes) -> None:
        assert looks_like_svg(document) is True

    @pytest.mark.parametrize(
        "data",
        [
            b"",
            b"this is not an image",
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
            b"\xff\xd8\xff\xe0\x00\x10JFIF",
            b"<html><body>hello</body></html>",
            # A tag whose name merely starts with "svg" is not the SVG root.
            b"<svgfoo></svgfoo>",
        ],
    )
    def test_does_not_claim_other_bytes(self, data: bytes) -> None:
        assert looks_like_svg(data) is False

    def test_a_mention_inside_a_comment_is_only_a_hint(self) -> None:
        """The hint may be wrong; ``inspect_svg`` is what actually decides."""
        assert looks_like_svg(b'<!-- <svg width="1" height="1"/> --><html/>') is True
        with pytest.raises(NotSvgDocumentError):
            inspect_svg(b'<!-- <svg width="1" height="1"/> --><html/>', max_depth=256)


class TestInspection:
    def test_reads_declared_geometry(self) -> None:
        doc = inspect_svg(
            f'<svg xmlns="{NS}" width="120" height="60" viewBox="0 0 240 120"/>'.encode(),
            max_depth=256,
        )
        assert doc.width == 120
        assert doc.height == 60
        assert doc.view_box == (0.0, 0.0, 240.0, 120.0)
        assert doc.declared_size == (120.0, 60.0)

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [("96px", 96.0), ("72pt", 96.0), ("1in", 96.0), ("25.4mm", 96.0), ("2.54cm", 96.0), ("1pc", 16.0)],
    )
    def test_absolute_lengths_convert_to_pixels(self, declared: str, expected: float) -> None:
        doc = inspect_svg(f'<svg xmlns="{NS}" width="{declared}" height="10"/>'.encode(), max_depth=256)
        assert doc.width == pytest.approx(expected)

    @pytest.mark.parametrize("declared", ["100%", "2em", "1.5rem", "10ex", "auto", "-5", "0", "1e999", ""])
    def test_relative_or_nonsense_lengths_are_unknown(self, declared: str) -> None:
        """A percentage depends on a viewport nobody supplied; guessing would be worse."""
        doc = inspect_svg(f'<svg xmlns="{NS}" width="{declared}" height="10"/>'.encode(), max_depth=256)
        assert doc.width is None

    def test_a_missing_axis_follows_the_viewbox_aspect(self) -> None:
        doc = inspect_svg(f'<svg xmlns="{NS}" width="100" viewBox="0 0 200 50"/>'.encode(), max_depth=256)
        assert doc.declared_size == (100.0, 25.0)

    def test_viewbox_is_the_fallback_when_no_size_is_declared(self) -> None:
        doc = inspect_svg(f'<svg xmlns="{NS}" viewBox="0 0 40 20"/>'.encode(), max_depth=256)
        assert doc.declared_size == (40.0, 20.0)

    def test_nothing_declared_means_nothing_known(self) -> None:
        doc = inspect_svg(f'<svg xmlns="{NS}"><rect/></svg>'.encode(), max_depth=256)
        assert doc.declared_size is None

    def test_illustrator_doctype_and_entities_are_tolerated(self) -> None:
        doc = inspect_svg(ILLUSTRATOR_SVG.encode(), max_depth=256)
        assert doc.declared_size == (120.0, 60.0)

    def test_an_external_dtd_is_never_fetched(self) -> None:
        """``SYSTEM``/``PUBLIC`` identifiers are recorded by expat, not resolved."""
        document = (
            b'<?xml version="1.0"?>'
            b'<!DOCTYPE svg SYSTEM "http://127.0.0.1:1/evil.dtd">'
            b'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"/>'
        )
        assert inspect_svg(document, max_depth=256).width == 8

    def test_an_external_entity_is_not_expanded(self) -> None:
        """If it were, the file's contents would end up in the rendered document."""
        document = (
            b'<?xml version="1.0"?>'
            b'<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            b'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><text>&xxe;</text></svg>'
        )
        assert inspect_svg(document, max_depth=256).width == 8

    def test_billion_laughs_is_refused_quickly(self) -> None:
        """expat's own amplification guard, reached before any tree is built."""
        entities = '<!ENTITY a0 "aaaaaaaaaa">'
        for level in range(1, 10):
            entities += f'<!ENTITY a{level} "' + f"&a{level - 1};" * 10 + '">'
        document = (
            f'<?xml version="1.0"?><!DOCTYPE svg [{entities}]><svg xmlns="{NS}" width="8" height="8">&a9;</svg>'
        ).encode()
        with pytest.raises(UnsupportedFormatError):
            inspect_svg(document, max_depth=256)

    def test_deep_nesting_is_refused_without_recursing(self) -> None:
        """A tree-building parser would raise RecursionError here; this refuses."""
        with pytest.raises(LimitExceededError) as excinfo:
            inspect_svg(deep_document(50_000), max_depth=256)
        assert excinfo.value.details["limit"] == 256

    def test_malformed_xml_reports_where(self) -> None:
        with pytest.raises(UnsupportedFormatError) as excinfo:
            inspect_svg(b'<svg xmlns="http://www.w3.org/2000/svg"><rect', max_depth=256)
        assert "line" in excinfo.value.details and "column" in excinfo.value.details

    def test_a_non_svg_root_is_not_an_svg_problem(self) -> None:
        with pytest.raises(NotSvgDocumentError):
            inspect_svg(b"<html><body/></html>", max_depth=256)

    def test_a_prefixed_root_is_recognised(self) -> None:
        doc = inspect_svg(f'<svg:svg xmlns:svg="{NS}" width="8" height="8"/>'.encode(), max_depth=256)
        assert doc.width == 8


class TestRenderTarget:
    @staticmethod
    def _policy(**overrides: object) -> SvgPolicy:
        return SvgPolicy(**overrides)  # type: ignore[arg-type]

    def test_uses_the_declared_size(self) -> None:
        doc = inspect_svg(f'<svg xmlns="{NS}" width="120" height="60"/>'.encode(), max_depth=256)
        assert render_target(doc, self._policy(), max_dimension=24_000, max_pixels=64_000_000) == (120, 60, False)

    def test_falls_back_to_the_configured_canvas(self) -> None:
        doc = inspect_svg(f'<svg xmlns="{NS}"/>'.encode(), max_depth=256)
        width, height, reduced = render_target(
            doc, self._policy(default_size=512), max_dimension=24_000, max_pixels=64_000_000
        )
        assert (width, height) == (512, 512)
        assert reduced is False

    def test_an_enormous_declaration_is_clamped_not_obeyed(self) -> None:
        """The defence against a 1000000x1000000 claim is this decision, not the renderer."""
        doc = inspect_svg(f'<svg xmlns="{NS}" width="1000000" height="1000000"/>'.encode(), max_depth=256)
        width, height, reduced = render_target(doc, self._policy(), max_dimension=24_000, max_pixels=64_000_000)
        assert width * height <= 64_000_000
        assert width == height == 8_000  # sqrt(64e6)
        assert reduced is True

    def test_the_per_axis_limit_wins_for_a_long_thin_document(self) -> None:
        doc = inspect_svg(f'<svg xmlns="{NS}" width="100000" height="10"/>'.encode(), max_depth=256)
        width, height, reduced = render_target(doc, self._policy(), max_dimension=24_000, max_pixels=64_000_000)
        assert width == 24_000 and height == 2
        assert reduced is True

    def test_aspect_ratio_is_preserved(self) -> None:
        doc = inspect_svg(f'<svg xmlns="{NS}" width="4000" height="1000"/>'.encode(), max_depth=256)
        width, height, _ = render_target(doc, self._policy(), max_dimension=1000, max_pixels=64_000_000)
        assert (width, height) == (1000, 250)

    def test_a_non_finite_declaration_cannot_produce_a_zero_target(self) -> None:
        doc = inspect_svg(f'<svg xmlns="{NS}" width="1e309" height="1e309"/>'.encode(), max_depth=256)
        width, height, _ = render_target(
            doc, self._policy(default_size=64), max_dimension=24_000, max_pixels=64_000_000
        )
        assert (width, height) == (64, 64)


@pytest.mark.skipif(not renderer_available(), reason="resvg-py is not installed")
class TestRasterisation:
    async def test_renders_an_illustrator_file(self, loader: ImageLoader, sandbox: Sandbox) -> None:
        loaded = await loader.load(ImageSource.parse(path="logo.svg"))
        try:
            assert loaded.geometry.width == 120
            assert loaded.geometry.height == 60
            assert loaded.original_format == "svg"
            assert loaded.mime_type == "image/svg+xml"
            assert loaded.image.mode == "RGBA"
            # The rasterisation is reported, not silent.
            assert any("rasterised" in note for note in loaded.notes), loaded.notes
        finally:
            loaded.close()

    async def test_accepts_a_viewbox_only_document(self, loader: ImageLoader) -> None:
        loaded = await loader.load(_inline(svg_document('<rect width="40" height="20"/>', attrs='viewBox="0 0 40 20"')))
        try:
            assert (loaded.geometry.width, loaded.geometry.height) == (40, 20)
        finally:
            loaded.close()

    async def test_paints_the_declared_colours(self, loader: ImageLoader) -> None:
        """A render that returns the right size but no pixels is not a success."""
        document = svg_document('<rect width="16" height="16" fill="#ff0000"/>')
        loaded = await loader.load(_inline(document))
        try:
            red, green, blue, alpha = loaded.image.convert("RGBA").getpixel((8, 8))
            assert (red, green, blue) == (255, 0, 0)
            assert alpha == 255
        finally:
            loaded.close()

    async def test_a_document_with_no_size_gets_the_default_canvas(self, loader: ImageLoader, config: Config) -> None:
        loaded = await loader.load(_inline(f'<svg xmlns="{NS}"><circle cx="5" cy="5" r="5"/></svg>'.encode()))
        try:
            assert loaded.geometry.width == config.svg.default_size
        finally:
            loaded.close()

    async def test_an_enormous_declaration_is_downscaled(self, loader: ImageLoader, config: Config) -> None:
        loaded = await loader.load(
            _inline(f'<svg xmlns="{NS}" width="1000000" height="1000000"><rect width="1" height="1"/></svg>'.encode())
        )
        try:
            assert loaded.geometry.pixels <= config.limits.max_pixels
            assert any("reduced" in note for note in loaded.notes), loaded.notes
        finally:
            loaded.close()

    async def test_source_bytes_are_reported_not_the_render(self, loader: ImageLoader) -> None:
        document = svg_document('<rect width="16" height="16"/>')
        loaded = await loader.load(_inline(document))
        try:
            assert loaded.byte_size == len(document)
        finally:
            loaded.close()


#: A program that kills its own process with SIGSEGV - the way an overflowing
#: renderer dies - so the crash-handling path can be tested on any host, whatever
#: that host's stack size happens to be. Core dumps are disabled first (where the
#: platform has ``resource``) so a failing run cannot litter the working directory.
_CRASHING_CHILD = (
    "import os, signal\n"
    "try:\n"
    "    import resource\n"
    "\n"
    "    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
    "except ImportError:\n"
    "    pass\n"
    "os.kill(os.getpid(), signal.SIGSEGV)\n"
)


@pytest.mark.skipif(not renderer_available(), reason="resvg-py is not installed")
class TestHostileInputCannotTakeTheServerDown:
    """The regression tests for the crash that forced the subprocess design.

    The renderer's breaking point is a property of the stack it runs on, not of the
    document, so it does not reproduce the same way everywhere: the 300-element
    ``<pattern>`` chain below overflows the stack on macOS arm64 and renders happily
    on glibc x86_64, which is where CI runs. A test that insists on the crash is
    therefore asserting a platform - and, worse, it leaves this file's most
    important property, that a dying child is survivable, covered by nothing at all
    on the very platform the suite runs on in CI.

    So the crash path is pinned by injecting a child that kills itself with SIGSEGV,
    which is deterministic everywhere, and the real document is exercised separately
    for whatever the host happens to do with it.
    """

    @staticmethod
    def _suicidal_child(policy: SvgPolicy, width: int, height: int) -> list[str]:
        """A child that dies on SIGSEGV, the way an overflowing renderer does.

        Core dumps are disabled first, so a failing run cannot litter the working
        directory with a core file.
        """
        del policy, width, height
        return [sys.executable, "-c", _CRASHING_CHILD]

    async def test_a_child_killed_by_a_signal_becomes_a_refused_call(
        self, loader: ImageLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exit-by-signal is precisely the case that cannot be caught in-process."""
        monkeypatch.setattr(vector, "_child_command", self._suicidal_child)
        with pytest.raises(UnsupportedFormatError) as excinfo:
            await loader.load(_inline(svg_document('<rect width="16" height="16"/>')))

        assert "could not be rasterised" in excinfo.value.message
        assert excinfo.value.details["detected_format"] == "svg"
        # Whatever the child said on its way out stays in the server log.
        assert "resvg" not in excinfo.value.message

    async def test_a_child_that_dies_does_not_poison_the_loader(
        self, loader: ImageLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real assertion: the next request must still work."""
        with monkeypatch.context() as patched:
            patched.setattr(vector, "_child_command", self._suicidal_child)
            with pytest.raises(UnsupportedFormatError):
                await loader.load(_inline(svg_document('<rect width="16" height="16"/>')))

        loaded = await loader.load(_inline(svg_document('<rect width="16" height="16" fill="#00ff00"/>')))
        try:
            assert (loaded.geometry.width, loaded.geometry.height) == (16, 16)
            assert loaded.image.convert("RGBA").getpixel((8, 8))[:3] == (0, 255, 0)
        finally:
            loaded.close()

    async def test_deep_nesting_is_refused_before_the_renderer_sees_it(self, loader: ImageLoader) -> None:
        """The depth ceiling exists so this input never reaches the renderer at all."""
        with pytest.raises(LimitExceededError):
            await loader.load(_inline(deep_document(1_000)))

    @pytest.mark.parametrize("count", [25, 200, 300])
    async def test_a_real_document_at_the_renderers_breaking_point(self, loader: ImageLoader, count: int) -> None:
        """Chain lengths below, at and past the point where the stack gives out.

        300 crashes on macOS arm64 and renders on glibc x86_64, so the outcome is
        whatever the host does - both branches are exercised here, because 25 and 200
        render on either platform. What is not acceptable is the server dying, or the
        caller being told about it in a library's words.
        """
        try:
            loaded = await loader.load(_inline(pattern_chain(count)))
        except PictorError as exc:
            assert exc.code in {"unsupported_format", "limit_exceeded"}
            assert "resvg" not in exc.message
        else:
            try:
                assert loaded.geometry.pixels == 16 * 16
            finally:
                loaded.close()

        # Whichever way the host went, the loader is still usable afterwards.
        reopened = await loader.load(_inline(svg_document('<rect width="16" height="16"/>')))
        try:
            assert reopened.geometry.pixels == 16 * 16
        finally:
            reopened.close()

    async def test_an_external_file_reference_is_not_read(self, loader: ImageLoader) -> None:
        """If the reference were followed, the pixels would carry the file's data."""
        document = (
            f'<svg xmlns="{NS}" xmlns:xlink="{XLINK}" width="32" height="32">'
            f'<image xlink:href="file:///etc/passwd" width="32" height="32"/></svg>'
        ).encode()
        loaded = await loader.load(_inline(document))
        try:
            alpha = loaded.image.convert("RGBA").getchannel("A")
            assert alpha.getextrema() == (0, 0), "the referenced file was rendered into the image"
        finally:
            loaded.close()

    async def test_an_http_reference_is_not_fetched(self, loader: ImageLoader) -> None:
        document = (
            f'<svg xmlns="{NS}" xmlns:xlink="{XLINK}" width="32" height="32">'
            f'<image xlink:href="http://169.254.169.254/latest/meta-data/" width="32" height="32"/></svg>'
        ).encode()
        loaded = await loader.load(_inline(document))
        try:
            assert loaded.image.convert("RGBA").getchannel("A").getextrema() == (0, 0)
        finally:
            loaded.close()

    async def test_script_is_inert(self, loader: ImageLoader) -> None:
        """There is no script engine: the element is parsed and ignored."""
        document = svg_document('<script>throw new Error("boom")</script><rect width="16" height="16" fill="#00ff00"/>')
        loaded = await loader.load(_inline(document))
        try:
            assert loaded.image.convert("RGBA").getpixel((8, 8))[:3] == (0, 255, 0)
        finally:
            loaded.close()

    async def test_an_unresolvable_entity_reference_is_refused(self, loader: ImageLoader) -> None:
        with pytest.raises(UnsupportedFormatError):
            await loader.load(_inline(f'<svg xmlns="{NS}" width="8" height="8">&nope;</svg>'.encode()))


class TestPolicy:
    """Switching the feature off, and the renderer being absent, are both explicit."""

    async def test_disabled_svg_says_so_and_names_the_switch(self, sandbox: Sandbox) -> None:
        config = load_config(_env(sandbox, PICTOR_ALLOW_SVG="false"))
        loader = _loader(config)
        with pytest.raises(UnsupportedFormatError) as excinfo:
            await loader.load(_inline(svg_document('<rect width="8" height="8"/>')))
        assert "PICTOR_ALLOW_SVG" in excinfo.value.message
        assert excinfo.value.details["detected_format"] == "svg"

    async def test_a_missing_renderer_names_the_package(
        self, loader: ImageLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vector, "renderer_version", lambda: None)
        with pytest.raises(UnsupportedFormatError) as excinfo:
            await loader.load(_inline(svg_document('<rect width="8" height="8"/>')))
        assert "resvg-py" in excinfo.value.message

    async def test_a_timeout_is_a_limit_not_a_crash(self, sandbox: Sandbox) -> None:
        """One microsecond is below the cost of starting the child, every time."""
        policy = SvgPolicy(timeout_seconds=1e-6)
        with pytest.raises(LimitExceededError):
            vector.rasterize_svg(
                svg_document('<rect width="16" height="16"/>'),
                policy,
                max_dimension=24_000,
                max_pixels=64_000_000,
            )

    def test_disabled_svg_is_not_advertised_as_readable(self) -> None:
        catalogue = public_format_catalogue(svg_enabled=False)
        assert "svg" not in {entry["format"] for entry in catalogue["readable"]}

    def test_svg_is_never_a_writable_format(self) -> None:
        """An image server that writes SVG is a stored-XSS primitive."""
        assert "svg" not in OUTPUT_FORMATS
        with pytest.raises(UnsupportedFormatError):
            resolve_output_format("svg")

    def test_svg_is_registered_as_readable(self) -> None:
        assert INPUT_FORMATS["svg"].mime == "image/svg+xml"
        assert INPUT_FORMATS["svg"].extension == ".svg"

    def test_the_catalogue_lists_svg_only_when_it_can_work(self) -> None:
        formats = {entry["format"] for entry in public_format_catalogue()["readable"]}
        assert ("svg" in formats) is renderer_available()


class TestNonSvgBytesStillReachPillow:
    """The prefix hint must never hijack a raster file that merely mentions SVG."""

    async def test_html_that_mentions_svg_is_still_not_an_image(self, loader: ImageLoader) -> None:
        with pytest.raises(UnsupportedFormatError) as excinfo:
            await loader.load(_inline(b'<!-- <svg width="8" height="8"/> --><html><body>hi</body></html>'))
        assert "not a recognisable image" in excinfo.value.message

    async def test_a_jpeg_with_svg_in_its_metadata_still_decodes(self, loader: ImageLoader, sandbox: Sandbox) -> None:
        """The hint fires, the parse says "not an SVG", and Pillow gets the bytes."""
        source = sandbox.inputs("photo.jpg")
        image = Image.open(source)
        try:
            image.save(
                sandbox.inputs("commented.jpg"),
                comment=b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>',
            )
        finally:
            image.close()

        raw = sandbox.inputs("commented.jpg").read_bytes()
        assert looks_like_svg(raw) is True, "the fixture no longer exercises the fallback"

        loaded = await loader.load(_inline(raw))
        try:
            assert loaded.original_format == "jpeg"
            assert loaded.notes == []
        finally:
            loaded.close()

    async def test_a_truncated_svg_reports_a_parse_error(self, loader: ImageLoader) -> None:
        with pytest.raises(UnsupportedFormatError) as excinfo:
            await loader.load(_inline(f'<svg xmlns="{NS}" width="8" height="8"><rect'.encode()))
        assert "could not be parsed" in excinfo.value.message


class TestOutput:
    async def test_converting_an_svg_defaults_to_png(self, loader: ImageLoader, sandbox: Sandbox) -> None:
        from pictor_mcp.tools.basic import _output_spec_key

        loaded = await loader.load(ImageSource.parse(path="logo.svg"))
        try:
            key, notes = _output_spec_key(loaded, None)
            assert key == "png"
            assert any("cannot be written" in note for note in notes), notes
            assert default_filename(loaded, resolve_output_format("png")) == "logo.png"
        finally:
            loaded.close()


class TestRendererChildContract:
    """The child is a separate program, so its protocol is worth pinning directly."""

    def _child(self, *args: str, stdin: bytes, flags: tuple[str, ...] = ("-B",)) -> subprocess.CompletedProcess[bytes]:
        source_root = Path(__file__).resolve().parent.parent / "src"
        env = {"PYTHONPATH": str(source_root), "PATH": "/usr/bin:/bin"}
        return subprocess.run(
            [sys.executable, *flags, "-m", "pictor_mcp.imaging._svgrender", *args],
            input=stdin,
            capture_output=True,
            env=env,
            timeout=60,
            check=False,
        )

    @pytest.mark.skipif(not renderer_available(), reason="resvg-py is not installed")
    def test_renders_png_on_stdout(self) -> None:
        result = self._child("--width", "16", "--height", "16", stdin=svg_document('<rect width="16" height="16"/>'))
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith(b"\x89PNG\r\n\x1a\n")

    def test_refuses_a_non_positive_size(self) -> None:
        result = self._child("--width", "0", "--height", "16", stdin=b"<svg/>")
        assert result.returncode != 0
        assert result.stdout == b""

    @pytest.mark.skipif(not renderer_available(), reason="resvg-py is not installed")
    def test_a_bad_document_fails_without_writing_stdout(self) -> None:
        result = self._child("--width", "16", "--height", "16", stdin=b"not an svg at all")
        assert result.returncode != 0
        assert result.stdout == b""

    @pytest.mark.skipif(not renderer_available(), reason="resvg-py is not installed")
    def test_an_unloadable_renderer_gets_its_own_exit_code(self) -> None:
        """``-S`` hides site-packages, so resvg_py exists but cannot be imported.

        The distinct exit code is what lets the parent say "install the dependency"
        rather than blaming the caller's document.
        """
        result = self._child(
            "--width",
            "16",
            "--height",
            "16",
            stdin=svg_document('<rect width="16" height="16"/>'),
            flags=("-B", "-S"),
        )
        assert result.returncode == 5, result.stderr
        assert result.stdout == b""

    @pytest.mark.skipif(not renderer_available(), reason="resvg-py is not installed")
    def test_system_fonts_can_be_switched_off(self) -> None:
        document = svg_document('<text x="1" y="12" font-size="12">Hi</text>')
        result = self._child("--width", "16", "--height", "16", "--no-system-fonts", stdin=document)
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith(b"\x89PNG\r\n\x1a\n")


class TestChildIsolation:
    """The child parses hostile input, so it must not be handed the server's secrets."""

    def test_the_environment_carries_no_server_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PICTOR_AUTH_TOKEN", "a-sufficiently-long-secret-token")
        monkeypatch.setenv("PICTOR_URL_SECRET", "another-sufficiently-long-secret")
        monkeypatch.setenv("PICTOR_OUTPUT_ROOT", "/data/output")

        env = vector._child_environment()

        assert not [key for key in env if key.startswith("PICTOR_")], sorted(env)
        # The package parent still comes first, so a source checkout works.
        package_parent = str(Path(vector.__file__).resolve().parents[2])
        assert env["PYTHONPATH"].split(os.pathsep)[0] == package_parent

    def test_startup_hooks_are_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``PYTHONSTARTUP`` would run an arbitrary file in the child interpreter."""
        monkeypatch.setenv("PYTHONSTARTUP", "/tmp/should-not-run.py")
        monkeypatch.setenv("PYTHONINSPECT", "1")
        env = vector._child_environment()
        assert "PYTHONSTARTUP" not in env
        assert "PYTHONINSPECT" not in env

    @pytest.mark.skipif(not renderer_available(), reason="resvg-py is not installed")
    async def test_a_render_still_works_while_secrets_are_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stripping the environment must not break the child it is meant to protect."""
        monkeypatch.setenv("PICTOR_AUTH_TOKEN", "a-sufficiently-long-secret-token")
        image, _notes = vector.rasterize_svg(
            svg_document('<rect width="16" height="16" fill="#0000ff"/>'),
            SvgPolicy(),
            max_dimension=24_000,
            max_pixels=64_000_000,
        )
        try:
            assert image.size == (16, 16)
        finally:
            image.close()


def _inline(document: bytes) -> ImageSource:
    return ImageSource(kind="base64", value=base64.b64encode(document).decode())


def _loader(config: Config) -> ImageLoader:
    jail = PathJail(config.input_roots, config.output_root)
    jail.ensure_output_root()
    return ImageLoader(config, jail, SafeFetcher(config.fetch))


def _env(sandbox: Sandbox, **overrides: str) -> dict[str, str]:
    env = {
        "PICTOR_INPUT_ROOTS": str(sandbox.input),
        "PICTOR_OUTPUT_ROOT": str(sandbox.output),
        "PICTOR_LOG_LEVEL": "ERROR",
    }
    env.update(overrides)
    return env

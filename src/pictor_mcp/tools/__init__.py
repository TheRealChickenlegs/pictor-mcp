"""Tool registration.

Tools are grouped by intent rather than by implementation module, and every
group is registered onto the same server instance so a client sees one flat,
predictably named tool set.
"""

from __future__ import annotations

import logging

from .context import ToolContext

logger = logging.getLogger(__name__)


def register_all(server, ctx: ToolContext) -> list[str]:
    """Register every tool group; returns the registered tool names."""
    from . import analysis, basic, compose

    basic.register(server, ctx)
    compose.register(server, ctx)
    analysis.register(server, ctx)

    names = [
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
    ]
    logger.info("registered %d tools", len(names))
    return names


__all__ = ["ToolContext", "register_all"]

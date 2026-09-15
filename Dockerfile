# syntax=docker/dockerfile:1
#
# pictor-mcp container images.
#
# Four stages share one definition. Stage order matters twice over:
#
#   * a stage may only be based on an EARLIER stage, so `base` is defined first;
#   * a plain `docker build .` builds the LAST stage, so `default` is last and
#     is simply the CPU image again.
#
# Putting `base` last (so it is the default target) would make `FROM base AS gpu`
# a forward reference. Docker would not resolve that as a stage - it would try to
# PULL an image literally named `base` from a registry, which is both a broken
# build and a supply-chain hazard. Hence the explicit `default` alias stage.
#
#   base     -> CPU image. Small, non-root, hardened. `docker build .` gives you
#               this, via the `default` alias at the end of the file.
#   gpu      -> base + PyTorch CUDA wheels (the `[gpu]` extra).
#   ml       -> gpu + rembg/onnxruntime-gpu with the u2net weights baked in, so
#               the container needs no network at runtime.
#
# Build recipes:
#   docker build .                                       # CPU (default)
#   docker build --target gpu -t pictor-mcp:gpu .
#   docker build --target ml  -t pictor-mcp:ml  .
#
# Only `gpu` and `ml` need the network at BUILD time, and only to fetch large
# wheels and model weights. Every variant runs offline afterwards.
#
# Nothing writes to the root filesystem at runtime: the only writable paths are
# the /data/output volume and the /tmp tmpfs the operator supplies. See
# docker-compose.yml, which enforces that with read_only + tmpfs.


# =============================================================================
# base - the CPU image. This is what `docker build .` produces, via `default`.
# =============================================================================
FROM python:3.14-slim AS base

LABEL org.opencontainers.image.title="pictor-mcp" \
      org.opencontainers.image.description="Secure MCP server for image operations: convert, resize, compress, crop, watermark, batch and more." \
      org.opencontainers.image.source="https://github.com/TheRealChickenlegs/pictor-mcp" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Runtime shared libraries for the Pillow codecs this server uses:
#   fonts-dejavu-core -> a real font for the text watermark tool (see
#                        imaging/fonts.py; without it Pillow falls back to its
#                        unscalable bitmap default).
#   libimagequant0    -> palette quantisation for GIF/PNG output.
#   libwebp7          -> WebP encode/decode.
#   libjpeg62-turbo   -> JPEG encode/decode, including progressive JPEG.
#
# The package lists are removed in the same layer so the apt index neither
# bloats the image nor lingers as a stale cache for a later layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        libimagequant0 \
        libwebp7 \
        libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime identity: no shell, no home directory, high uid so it
# cannot collide with a host user when /data is bind-mounted.
RUN groupadd --system --gid 10001 pictor \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin pictor

WORKDIR /app

# The build context is pyproject.toml + README.md + src/ (see .dockerignore);
# nothing else is copied, so no secrets, tests or host-local files can reach a
# layer. README.md is needed because pyproject.toml declares `readme = ...`.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# `pip install .` installs the package and its console script. The /app/src copy
# stays in the image so the sources are inspectable and `python -m pictor_mcp`
# resolves exactly as it does from a checkout.
RUN pip install --no-cache-dir .

# Data layout. /data/input is root-owned and world-readable, so the unprivileged
# service cannot rewrite its own inputs. /data/output is owned by pictor with
# 0750, because paths.PathJail performs every write there and nowhere else (its
# atomic-write temp file is created in the destination directory, not /tmp).
RUN mkdir -p /data/input /data/output \
    && chown -R root:root /data/input \
    && chmod 0755 /data/input \
    && chown -R pictor:pictor /data/output \
    && chmod 0750 /data/output

# Defaults bind all interfaces *inside* the container, which is the only way the
# published port can reach it. Real reachability is decided by the compose port
# mapping, which is loopback-only. The image ships no credential: PICTOR_AUTH_TOKEN
# is supplied by the operator, so publishing this image leaks nothing.
ENV PICTOR_TRANSPORT=streamable-http \
    PICTOR_HOST=0.0.0.0 \
    PICTOR_PORT=8077 \
    PICTOR_INPUT_ROOTS=/data/input \
    PICTOR_OUTPUT_ROOT=/data/output \
    PICTOR_FONT_DIRS=/usr/share/fonts
# HOME points at the /tmp tmpfs: the runtime user has no home directory, and any
# library writing ~/.cache or ~/.config would otherwise fail on the read-only
# root filesystem. NUMBA_CACHE_DIR keeps native caches there too.
ENV HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp

USER pictor

EXPOSE 8077

# Documents the only directory this service ever writes, and keeps it writable
# even though compose mounts the root filesystem read-only.
VOLUME ["/data/output"]

# curl is deliberately not installed, so the probe uses the interpreter already
# present. /healthz is information-free and exempt from bearer auth (see
# server.py), so this works with or without PICTOR_AUTH_TOKEN.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8077/healthz', timeout=4).status == 200 else 1)"

# `python -m pictor_mcp` rather than the console script, so the container still
# works if an operator overrides PYTHONPATH or reinstalls the package.
ENTRYPOINT ["python", "-m", "pictor_mcp"]


# =============================================================================
# gpu - CPU base plus PyTorch CUDA wheels.
#
# TORCH_VERSION is a build argument so an operator can pin an exact build for
# reproducibility. Note that `--extra-index-url` is what PyTorch's own install
# instructions use, but it does expose pip's dependency-confusion surface: pip
# considers both PyPI and the CUDA index and takes the highest version, so a
# package squatting the `torch` name on PyPI with a higher version would win.
# Pinning the version (and building from a trusted network) is the mitigation;
# hash-pinning is impractical while pyproject.toml uses version ranges.
# The CPU image - the default target - never touches the CUDA index.
# =============================================================================
FROM base AS gpu

ARG TORCH_VERSION=2.4.1

# The base stage ends as the unprivileged `pictor` user; installation needs root.
USER root

RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        "torch==${TORCH_VERSION}" \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        ".[gpu]"

ENV PICTOR_GPU=auto

USER pictor


# =============================================================================
# ml - GPU image plus ML background removal, with u2net baked in.
#
# BUILD-TIME INTERNET IS REQUIRED FOR THIS STAGE ONLY: it downloads the
# onnxruntime-gpu wheel and the ~176 MB u2net.onnx weights from the rembg
# release assets. The resulting image is fully offline-capable, which is what
# makes it usable on an isolated internal network.
# =============================================================================
FROM gpu AS ml

USER root

# The extras are resolved as explicit packages rather than `".[bg,bg-gpu]"` so
# that `onnxruntime` and `onnxruntime-gpu` can never both be installed: they
# ship the same `onnxruntime` import package, and the CPU wheel would shadow the
# CUDA one. rembg 2.0.57+ does not depend on onnxruntime itself, so its other
# dependencies come from PyPI while the runtime comes from the GPU wheel here.
RUN pip install --no-cache-dir "onnxruntime-gpu>=1.18" "rembg>=2.0.57"

# onnxruntime and scipy (used by rembg's alpha matting) dlopen libgomp.so.1 at
# import time, and python:*-slim does not ship it. libglib2.0-0 is the one
# system library the headless OpenCV wheel occasionally needs. Both are a couple
# of megabytes; the alternative is an ImportError at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Bake the segmentation model into the image.
#
# U2NET_HOME takes precedence over the newer REMBG_HOME in rembg's model
# resolution and keeps the weights in a system location rather than $HOME, which
# matters because the runtime user has no writable home. `new_session()` is what
# triggers the download: it builds an ONNX session from the weights, fetching
# them first if absent.
#
# The resulting tree is /opt/models/models/u2net/u2net.onnx (rembg appends
# `models/<name>` to the configured home). It is root-owned and world-readable
# only, so a compromised runtime cannot swap the weights. It also means only the
# baked-in model works offline; any other PICTOR_BG_MODEL must be pre-seeded the
# same way at build time or mounted in.
ENV U2NET_HOME=/opt/models
RUN mkdir -p /opt/models \
    && U2NET_HOME=/opt/models python -c "from rembg import new_session; new_session('u2net')" \
    && chmod -R a+rX /opt/models

USER pictor


# =============================================================================
# default - an alias for the CPU image, placed last so that a plain
# `docker build .` produces the small, hardened image rather than the multi-
# gigabyte CUDA one.
# =============================================================================
FROM base AS default

# syntax=docker/dockerfile:1
#
# pictor-mcp container images.
#
# Seven stages share one definition, and the split is not cosmetic - it is what
# keeps a source edit from re-downloading gigabytes. Two rules govern the order:
#
#   * a stage may only be based on an EARLIER stage, so `system` is defined first;
#   * a plain `docker build .` builds the LAST stage, so `default` is last and
#     is simply the CPU image again.
#
# Putting `base` last (so it is the default target) would make `FROM base AS gpu`
# a forward reference. Docker would not resolve that as a stage - it would try to
# PULL an image literally named `base` from a registry, which is both a broken
# build and a supply-chain hazard. Hence the explicit `default` alias stage.
#
#   system     -> interpreter, system libraries, runtime identity.
#   base-deps  -> system + the third-party runtime dependencies, WITHOUT the
#                 application itself. See "Why the application is installed last".
#   gpu-deps   -> base-deps + PyTorch CUDA wheels (the `[gpu]` extra).
#   ml-deps    -> gpu-deps + rembg/onnxruntime-gpu, with the u2net weights baked
#                 in so the container needs no network at runtime.
#   base       -> base-deps + the application. CPU image. `docker build .` gives
#                 you this, via the `default` alias at the end of the file.
#   gpu        -> gpu-deps  + the application.
#   ml         -> ml-deps   + the application.
#   default    -> an alias for `base`.
#
# The published target names are unchanged: `--target base|gpu|ml` still produce
# the CPU, CUDA and ML images. Only the internal `*-deps` stages are new, and
# they are never a valid thing to run - they have no application in them.
#
# Why the application is installed last
# -------------------------------------
# A Docker layer is invalidated by any change to the layers beneath it, and for
# a long time `COPY src/` sat *below* the PyTorch install. Every source edit
# therefore invalidated the CUDA layer, and the next build re-downloaded and
# reinstalled several gigabytes of NVIDIA wheels - the single most expensive
# thing about working on this repository, and the reason `git commit` used to
# cost more than the code change did.
#
# Now `COPY src/` happens only in the three application stages, above every
# dependency layer. A source edit rebuilds one small layer and leaves the CUDA
# and model layers untouched, whether the build runs on a laptop, in CI, or
# through the Portainer build API.
#
# The dependency stages still need a package to install, because `pip install .`
# is what resolves the dependency set from pyproject.toml - a hand-written
# `pip install` list here would be a second copy of pyproject.toml, and the two
# would drift. So they install against an empty placeholder package that the
# application stages then overwrite. That is why the application stages use
# `--force-reinstall`, and why they end with an import check: if the placeholder
# ever survived, the image would start and fail on an empty package, and that is
# a worse failure than a failed build.
#
# Build recipes:
#   docker build .                                       # CPU (default)
#   docker build --target gpu -t pictor-mcp:gpu .
#   docker build --target ml  -t pictor-mcp:ml  .
#
# A pip cache mount is attached to every dependency install. On a host with a
# persistent build cache (any normal Docker daemon) that means even a rebuild of
# the CUDA layer - after a pyproject.toml change, say - reuses the wheels already
# on disk instead of downloading them again. The mount is never written into an
# image layer, so no cache bloat ships.
#
# Only `gpu` and `ml` need the network at BUILD time, and only to fetch large
# wheels and model weights. Every variant runs offline afterwards.
#
# Nothing writes to the root filesystem at runtime: the only writable paths are
# the /data/output volume and the /tmp tmpfs the operator supplies. See
# docker-compose.yml, which enforces that with read_only + tmpfs.


# =============================================================================
# system - interpreter, system libraries and the runtime identity.
#
# Nothing here depends on the project, so nothing here is invalidated by a code
# change. Everything downstream inherits it, including all three variants.
# =============================================================================
FROM python:3.14-slim AS system

LABEL org.opencontainers.image.title="pictor-mcp" \
      org.opencontainers.image.description="Secure MCP server for image operations: convert, resize, compress, crop, watermark, batch and more." \
      org.opencontainers.image.source="https://github.com/TheRealChickenlegs/pictor-mcp" \
      org.opencontainers.image.licenses="MIT"

# PIP_ROOT_USER_ACTION silences pip's "running as root" advice: installing as
# root during the build is intended, because the runtime user does not exist yet.
# PIP_NO_CACHE_DIR=1 is the default for every install, so a `pip install` that
# forgets its cache mount cannot bloat a layer. The dependency installs below
# deliberately override it, because they mount a cache instead.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

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

# HOME points at the /tmp tmpfs: the runtime user has no home directory, and any
# library writing ~/.cache or ~/.config would otherwise fail on the read-only
# root filesystem. NUMBA_CACHE_DIR keeps native caches there too.
#
# Set here rather than after the dependency installs, and paired with an explicit
# PIP_CACHE_DIR below, so the build cache location never depends on HOME.
ENV HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp


# =============================================================================
# base-deps - the third-party runtime dependencies, and not the application.
#
# This is the layer the whole caching strategy rests on: it must not change when
# src/ changes. Anything added here is inherited by the CUDA and ML stages, so a
# mistake here costs a multi-gigabyte rebuild on every commit.
# =============================================================================
FROM system AS base-deps

# The build context is pyproject.toml + README.md + src/ (see .dockerignore);
# nothing else is copied, so no secrets, tests or host-local files can reach a
# layer. README.md is needed because pyproject.toml declares `readme = ...`.
COPY pyproject.toml README.md ./

# The placeholder package: an empty module, created here and gone by the time the
# image is finished. It exists so that `pip install .` can resolve the dependency
# list from pyproject.toml - the alternative, listing the dependencies again in
# this file, is the classic way the two copies drift apart.
#
# The pip cache mount is keyed to /opt/pip-cache and given a matching
# PIP_CACHE_DIR, so it does not depend on HOME (which is /tmp above) or on the
# build running as any particular user.
RUN --mount=type=cache,target=/opt/pip-cache \
    mkdir -p src/pictor_mcp \
    && touch src/pictor_mcp/__init__.py \
    && PIP_NO_CACHE_DIR=0 PIP_CACHE_DIR=/opt/pip-cache pip install .

# Data layout.
#
# /data/input is root-owned and world-readable, so the service cannot rewrite its
# own inputs.
#
# /data/output is deliberately writable by *any* uid (1777, the /tmp convention:
# anyone may create files, only the owner may remove them). The image's own user
# is `pictor` (10001), but operators routinely run the container as their own
# uid so that files land in ./output owned by them rather than by a stranger -
# docker-compose.yml does exactly that via PUID/PGID. A mode that only admitted
# 10001 would make that fail at startup, and with a bind mount the host
# directory's permissions govern anyway. The sticky bit keeps the two cases
# consistent rather than merely permissive.
#
# paths.PathJail performs every write here and nowhere else: its atomic-write
# temp file is created in the destination directory, not /tmp.
RUN mkdir -p /data/input /data/output \
    && chown -R root:root /data/input \
    && chmod 0755 /data/input \
    && chown -R root:root /data/output \
    && chmod 1777 /data/output

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

# The image's default identity. Non-root, with no shell and no home directory.
# Operators who want output files owned by their own host user override this with
# `user:` in compose (see PUID/PGID in .env.example); the /data/output mode above
# is what makes that work.
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
# gpu-deps - base-deps plus PyTorch CUDA wheels.
#
# The CUDA index and the base image's Python version are coupled, and that is the
# trap this stage fell into: a Dependabot bump moved the base from python:3.12 to
# python:3.14, and the pinned torch==2.4.1 then could not resolve, because the
# cu124 index has no cp314 wheels at all (it stops at torch 2.6.0). A hard pin
# plus a moving interpreter is a build that breaks on someone else's schedule.
#
# Two build arguments instead of one hard pin:
#
#   TORCH_INDEX_URL  which CUDA build to install. cu128 is the default: it is the
#                    first CUDA line with Blackwell kernels (sm_120, the RTX 50
#                    series), it still covers Turing through Hopper, and it
#                    publishes wheels for every interpreter the base might use,
#                    3.14 included. cu126 is NOT a viable default despite being
#                    older and therefore driver-friendlier - CUDA 12.6 predates
#                    Blackwell entirely, so a 50-series card would load torch and
#                    then fail every kernel launch.
#   TORCH_VERSION    empty means "the newest build on that index for this
#                    interpreter", which cannot go stale. Set it to pin exactly,
#                    for a reproducible image.
#
# Check the pairing before changing either:
#
#   curl -s https://download.pytorch.org/whl/cu128/torch/ | grep -o 'cp3[0-9]*' | sort -u
#
# The GPU's compute capability must also be in the chosen build's kernel list.
# The server reports this at startup when it is not, naming the capability and
# the architectures the build does support, so the mismatch is diagnosable
# without reading NVIDIA's documentation.
#
# If your driver is too old for a 12.8 CUDA build, pin an older index *and* a
# base image whose Python that index still publishes wheels for; note that
# anything below cu128 cannot drive a Blackwell card at all.
#
# The CPU image - the default target - never touches the CUDA index.
#
# This stage is the multi-gigabyte one, and it is deliberately below `COPY src/`
# in every chain that includes it. See the note at the top of this file.
# =============================================================================
FROM base-deps AS gpu-deps

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=

# The dependency stages end as the unprivileged `pictor` user; installation needs
# root.
USER root

# `--index-url` *replaces* PyPI rather than adding to it, and PyTorch's index
# mirrors torch's own dependencies (sympy, networkx, filelock, jinja2, fsspec),
# so this resolves completely. That is strictly better than the
# `--extra-index-url` form PyTorch documents: with two indexes in play, pip takes
# the highest version across both, which is the dependency-confusion opening.
# Here there is only ever one index.
# The second install is satisfied by the first and is kept so that adding a
# dependency to the `gpu` extra in pyproject.toml cannot be silently ignored
# here. It uses the default index, so it also cannot fail for lack of a wheel.
#
# Both installs share one cache mount, so the NVIDIA wheels this pulls down are
# reused by the next build on the same host even when this layer has to be
# rebuilt.
RUN --mount=type=cache,target=/opt/pip-cache \
    PIP_NO_CACHE_DIR=0 PIP_CACHE_DIR=/opt/pip-cache \
      pip install --index-url "${TORCH_INDEX_URL}" "torch${TORCH_VERSION:+==${TORCH_VERSION}}" \
    && PIP_NO_CACHE_DIR=0 PIP_CACHE_DIR=/opt/pip-cache \
      pip install ".[gpu]"

ENV PICTOR_GPU=auto

USER pictor


# =============================================================================
# ml-deps - gpu-deps plus ML background removal, with u2net baked in.
#
# BUILD-TIME INTERNET IS REQUIRED FOR THIS STAGE ONLY: it downloads the
# onnxruntime-gpu wheel and the ~176 MB u2net.onnx weights from the rembg
# release assets. The resulting image is fully offline-capable, which is what
# makes it usable on an isolated internal network.
# =============================================================================
FROM gpu-deps AS ml-deps

USER root

# The extras are resolved as explicit packages rather than `".[bg,bg-gpu]"` so
# that `onnxruntime` and `onnxruntime-gpu` can never both be installed: they
# ship the same `onnxruntime` import package, and the CPU wheel would shadow the
# CUDA one. rembg 2.0.57+ does not depend on onnxruntime itself, so its other
# dependencies come from PyPI while the runtime comes from the GPU wheel here.
RUN --mount=type=cache,target=/opt/pip-cache \
    PIP_NO_CACHE_DIR=0 PIP_CACHE_DIR=/opt/pip-cache \
      pip install "onnxruntime-gpu>=1.18" "rembg>=2.0.57"

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
# base, gpu, ml - the published images: dependencies plus the application.
#
# Each is two instructions' worth of work, and that is the point: `COPY src/`
# lives here, above every dependency layer, so a source edit rebuilds this layer
# and nothing else. Do not move a COPY or a pip install below this line into a
# dependency stage - that is precisely the mistake these stages exist to fix.
#
# `--force-reinstall --no-deps` replaces the placeholder package from base-deps.
# `--no-deps` keeps it from touching the dependency layers; `--force-reinstall`
# is what makes the replacement unconditional rather than something pip is
# allowed to decide it can skip.
#
# The import check is the guard rail: the placeholder module has no `__version__`
# and no `__main__`, so if the replacement ever stopped happening this build
# fails here, in a few seconds, instead of producing an image that dies on start.
# =============================================================================
FROM base-deps AS base

USER root
COPY src/ ./src/
RUN --mount=type=cache,target=/opt/pip-cache \
    PIP_NO_CACHE_DIR=0 PIP_CACHE_DIR=/opt/pip-cache \
      pip install --force-reinstall --no-deps . \
    && python -c "import pictor_mcp, pictor_mcp.server; print('pictor-mcp', pictor_mcp.__version__)"
USER pictor


FROM gpu-deps AS gpu

USER root
COPY src/ ./src/
RUN --mount=type=cache,target=/opt/pip-cache \
    PIP_NO_CACHE_DIR=0 PIP_CACHE_DIR=/opt/pip-cache \
      pip install --force-reinstall --no-deps . \
    && python -c "import pictor_mcp, pictor_mcp.server; print('pictor-mcp', pictor_mcp.__version__)"
USER pictor


FROM ml-deps AS ml

USER root
COPY src/ ./src/
RUN --mount=type=cache,target=/opt/pip-cache \
    PIP_NO_CACHE_DIR=0 PIP_CACHE_DIR=/opt/pip-cache \
      pip install --force-reinstall --no-deps . \
    && python -c "import pictor_mcp, pictor_mcp.server; print('pictor-mcp', pictor_mcp.__version__)"
USER pictor


# =============================================================================
# default - an alias for the CPU image, placed last so that a plain
# `docker build .` produces the small, hardened image rather than the multi-
# gigabyte CUDA one.
# =============================================================================
FROM base AS default

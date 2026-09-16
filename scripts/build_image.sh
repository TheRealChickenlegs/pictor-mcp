#!/usr/bin/env bash
#
# Build one pictor-mcp image into the local Docker daemon.
#
#   scripts/build_image.sh                      # CPU image,        pictor-mcp:local
#   scripts/build_image.sh --target gpu         # CUDA image,       pictor-mcp:local-gpu
#   scripts/build_image.sh --target ml          # CUDA + rembg,     pictor-mcp:local-ml
#
# Why this exists instead of a bare `docker build`:
#
#   * the tag follows the target, so a CUDA build cannot land under the CPU tag
#     and a running CPU stack cannot silently become a 6 GB CUDA one;
#   * the result is verified before you deploy it - the image is started with no
#     network, which is both the offline claim and the check that the
#     application layer really replaced the dependency stage's placeholder
#     package (see the note at the top of the Dockerfile);
#   * it prints the .env lines that deploy exactly this image, and the fact that
#     they exist is the whole point: nothing has to be re-downloaded, because
#     `pictor-mcp:local-gpu` is not in any registry.
#
# The dependency layers are cached by the Docker daemon, and every install is
# backed by a BuildKit cache mount, so a source edit rebuilds one small layer and
# a pyproject change reuses the wheels already on disk.
#
# Environment (all optional):
#   BUILD_TARGET        default target when --target is not given
#   BUILD_CACHE_FROM    buildx cache spec to read, e.g. type=registry,ref=...:buildcache
#   BUILD_CACHE_TO      buildx cache spec to write, for CI that has no local daemon
#   TORCH_INDEX_URL     passed through as a build arg for the gpu/ml targets
#   TORCH_VERSION       passed through as a build arg for the gpu/ml targets
#   DOCKER              docker binary to use (default: docker)

set -euo pipefail

DOCKER="${DOCKER:-docker}"
TARGET="${BUILD_TARGET:-default}"
TAG=""
PUSH=0
SMOKE=1
PRINT_ONLY=0
EXTRA_ARGS=()

usage() {
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
    printf 'build_image.sh: %s\n' "$*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ $# -ge 2 ]] || die "--target needs a value"
            TARGET="$2"
            shift 2
            ;;
        --tag)
            [[ $# -ge 2 ]] || die "--tag needs a value"
            TAG="$2"
            shift 2
            ;;
        --push)
            PUSH=1
            shift
            ;;
        --no-smoke)
            SMOKE=0
            shift
            ;;
        --build-arg)
            [[ $# -ge 2 ]] || die "--build-arg needs KEY=VALUE"
            EXTRA_ARGS+=(--build-arg "$2")
            shift 2
            ;;
        --print-command)
            # Prints what would run and exits, so the command can be asserted on
            # without a Docker daemon.
            PRINT_ONLY=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1 (try --help)"
            ;;
    esac
done

case "$TARGET" in
    default | base)
        TARGET="default"
        DEFAULT_TAG="local"
        ;;
    gpu)
        DEFAULT_TAG="local-gpu"
        ;;
    ml)
        DEFAULT_TAG="local-ml"
        ;;
    *)
        die "unknown target '$TARGET' (expected default, gpu or ml)"
        ;;
esac

TAG="${TAG:-pictor-mcp:${DEFAULT_TAG}}"
CONTEXT="${CONTEXT:-.}"

# The CUDA knobs are build arguments of the gpu stage. Passing them through the
# environment keeps the Dockerfile's defaults (newest torch on cu128) intact when
# they are unset.
[[ -n "${TORCH_INDEX_URL:-}" ]] && EXTRA_ARGS+=(--build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}")
[[ -n "${TORCH_VERSION:-}" ]] && EXTRA_ARGS+=(--build-arg "TORCH_VERSION=${TORCH_VERSION}")

# `docker buildx build --load` is what puts the result in the local daemon under
# a name other containers can use; without --load buildx keeps it in the build
# cache only. `docker build` is the fallback for a daemon with no buildx plugin,
# and still uses BuildKit (DOCKER_BUILDKIT=1) because the Dockerfile's cache
# mounts require it.
if [[ "$PUSH" == "1" ]]; then
    LOAD_ARGS=(--push)
else
    LOAD_ARGS=(--load)
fi

BUILD_CMD=("$DOCKER")
if "$DOCKER" buildx version >/dev/null 2>&1; then
    BUILD_CMD+=(buildx build)
else
    BUILD_CMD+=(build)
    export DOCKER_BUILDKIT=1
fi
BUILD_CMD+=(
    --target "$TARGET"
    --tag "$TAG"
    "${LOAD_ARGS[@]}"
)
[[ -n "${BUILD_CACHE_FROM:-}" ]] && BUILD_CMD+=(--cache-from "$BUILD_CACHE_FROM")
[[ -n "${BUILD_CACHE_TO:-}" ]] && BUILD_CMD+=(--cache-to "$BUILD_CACHE_TO")
BUILD_CMD+=("${EXTRA_ARGS[@]}" "$CONTEXT")

# No network: the image must be fully offline-capable at startup, and a
# placeholder package left behind by a broken application layer fails here
# rather than in the deployment.
SMOKE_CMD=("$DOCKER" run --rm --network none "$TAG" --version)

if [[ "$PRINT_ONLY" == "1" ]]; then
    printf '%q ' "${BUILD_CMD[@]}"
    printf '\n'
    printf '%q ' "${SMOKE_CMD[@]}"
    printf '\n'
    exit 0
fi

printf '==> building %s from target %s\n' "$TAG" "$TARGET"
"${BUILD_CMD[@]}"

if [[ "$SMOKE" == "1" ]]; then
    printf '==> verifying %s starts and reports its version, with no network\n' "$TAG"
    "${SMOKE_CMD[@]}"
fi

if [[ "$TARGET" == "default" ]]; then
    TAG_KEY="IMAGE_TAG"
else
    TAG_KEY="IMAGE_TAG_$(printf '%s' "$TARGET" | tr '[:lower:]' '[:upper:]')"
fi

if [[ "$TAG" == pictor-mcp:* ]]; then
    LOCAL_TAG="${TAG#pictor-mcp:}"
    cat <<EOF

Built $TAG.

To deploy it from a stack that names images rather than building them - a
Portainer stack, for instance - point the compose settings at this tag so
nothing is ever pulled:

    IMAGE_REPO=${TAG%%:*}
    ${TAG_KEY}=${LOCAL_TAG}
    LOCAL_IMAGE_TAG=${LOCAL_TAG}

The same values belong in .env next to docker-compose.yml; docs/portainer.md
covers the stack side.
EOF
else
    cat <<EOF

Built $TAG.

That name is not the local default, so if it lives in a registry rather than on
this host, add the registry in Portainer (Registries -> Add registry) and set
IMAGE_REPO / IMAGE_TAG on the stack to match. See docs/portainer.md.
EOF
fi

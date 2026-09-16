#!/usr/bin/env bash
#
# Build a pictor-mcp image with Portainer, on the Docker host Portainer manages.
#
#   PORTAINER_URL=https://portainer.internal:9443 \
#   PORTAINER_API_TOKEN=ptr_... \
#     scripts/portainer_build.sh --target gpu
#
# The build context is streamed to Portainer's Docker proxy
# (POST /api/endpoints/<id>/docker/build), so Portainer's own daemon builds the
# image - on the host that will run it. Nothing is pushed to a registry and
# nothing is pulled: the resulting tag exists only on that host, which is what
# makes this the cheap way to iterate on a multi-gigabyte CUDA image.
#
# What Portainer does *not* do is build on a git push. Its Workflows feature
# deploys compose files that reference an image; the image has to exist first.
# Run this from a git hook, from CI, or by hand - see docs/portainer.md.
#
# Only the files the Dockerfile reads are sent (Dockerfile, pyproject.toml,
# README.md, src/ and .dockerignore). That is deliberate: the context crosses the
# network, and a stray .env or output/ directory must not travel with it.
#
# Environment:
#   PORTAINER_URL          required, e.g. https://portainer.example.com:9443
#   PORTAINER_API_TOKEN    required, a Portainer access token (X-API-Key)
#   PORTAINER_ENDPOINT_ID  environment id, default 1
#   PORTAINER_INSECURE=1   skip TLS verification (self-signed certificate)
#   BUILD_TARGET           default target when --target is not given
#   TORCH_INDEX_URL        CUDA index to install from (gpu/ml targets)
#   TORCH_VERSION          exact torch version to install, empty for newest
#   CURL                   curl binary to use (default: curl)

set -euo pipefail

CURL="${CURL:-curl}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${BUILD_TARGET:-default}"
TAG=""
INSECURE="${PORTAINER_INSECURE:-0}"
NO_CACHE=0
VERIFY=1
PRINT_ONLY=0

#: Exactly what the build needs: the Dockerfile itself (the daemon resolves the
#: `dockerfile` parameter inside the context, so it must be in there even though
#: nothing COPYs it), every path the Dockerfile COPYs, and .dockerignore so the
#: exclusion rules travel with the context.
#: test_local_build.py fails if this drifts from the Dockerfile.
CONTEXT_PATHS=(Dockerfile pyproject.toml README.md src .dockerignore)

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
    printf 'portainer_build.sh: %s\n' "$*" >&2
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
        --insecure)
            INSECURE=1
            shift
            ;;
        --no-cache)
            NO_CACHE=1
            shift
            ;;
        --no-verify)
            VERIFY=0
            shift
            ;;
        --print-command)
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
    gpu) DEFAULT_TAG="local-gpu" ;;
    ml) DEFAULT_TAG="local-ml" ;;
    *) die "unknown target '$TARGET' (expected default, gpu or ml)" ;;
esac
TAG="${TAG:-pictor-mcp:${DEFAULT_TAG}}"

: "${PORTAINER_URL:?set PORTAINER_URL to the Portainer base URL, e.g. https://portainer:9443}"
: "${PORTAINER_API_TOKEN:?set PORTAINER_API_TOKEN to a Portainer access token}"
ENDPOINT="${PORTAINER_ENDPOINT_ID:-1}"
BASE="${PORTAINER_URL%/}/api/endpoints/${ENDPOINT}/docker"

#: Percent-encode the characters that appear in a JSON build-args object and
#: would otherwise break the query string. `=` is deliberately absent: it is
#: legal in a query value all by itself, and the values here are a URL and a
#: version number.
encode() {
    printf '%s' "$1" |
        sed -e 's/%/%25/g' \
            -e 's/"/%22/g' \
            -e 's/{/%7B/g' \
            -e 's/}/%7D/g' \
            -e 's/:/%3A/g' \
            -e 's|/|%2F|g' \
            -e 's/,/%2C/g' \
            -e 's/&/%26/g' \
            -e 's/?/%3F/g' \
            -e 's/#/%23/g' \
            -e 's/ /%20/g'
}

QUERY="t=${TAG}&target=${TARGET}&dockerfile=Dockerfile"
[[ "$NO_CACHE" == "1" ]] && QUERY="${QUERY}&nocache=1"
# The CUDA knobs are Docker build args. Portainer proxies the Docker API
# verbatim, so they go out as the `buildargs` query parameter - a JSON object,
# which has to be encoded here because curl is sending a tar as the body and so
# cannot use --data-urlencode for it.
if [[ -n "${TORCH_INDEX_URL:-}" || -n "${TORCH_VERSION:-}" ]]; then
    ARGS_JSON="{"
    SEP=""
    if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
        ARGS_JSON="${ARGS_JSON}${SEP}\"TORCH_INDEX_URL\":\"${TORCH_INDEX_URL}\""
        SEP=","
    fi
    if [[ -n "${TORCH_VERSION:-}" ]]; then
        ARGS_JSON="${ARGS_JSON}${SEP}\"TORCH_VERSION\":\"${TORCH_VERSION}\""
    fi
    ARGS_JSON="${ARGS_JSON}}"
    QUERY="${QUERY}&buildargs=$(encode "$ARGS_JSON")"
fi

# Docker's build endpoint answers with newline-delimited JSON: progress arrives
# as {"stream":"..."} and a failure as {"error":"..."}, usually with HTTP 200 -
# the request succeeded, the build did not. So the stream is the verdict, not the
# status code, and both are checked.
render_stream() {
    while IFS= read -r line; do
        case "$line" in
            *'"stream":"'*)
                chunk="${line#*\"stream\":\"}"
                chunk="${chunk%%\"*}"
                # %b turns the \n and \t the JSON carries back into real ones.
                # Docker also emits colour as \u001b[...m, which %b leaves alone,
                # so strip it rather than printing it at the reader.
                printf '%b' "$chunk" | sed 's/\\u001b\[[0-9;]*m//g'
                ;;
            *'"error"'*)
                printf 'portainer: %s\n' "$line" >&2
                ;;
            *) : ;;
        esac
    done
}

TAR_CMD=(tar -C "$REPO_ROOT" -cf - --exclude='__pycache__' --exclude='*.py[co]' "${CONTEXT_PATHS[@]}")
CURL_ARGS=(-N -sS -X POST "${BASE}/build?${QUERY}"
    -H "X-API-Key: ${PORTAINER_API_TOKEN}"
    -H "Content-Type: application/x-tar"
    --data-binary @-)
[[ "$INSECURE" == "1" ]] && CURL_ARGS+=(-k)

if [[ "$PRINT_ONLY" == "1" ]]; then
    printf 'curl %s\n' "${CURL_ARGS[*]}"
    printf 'body: %s\n' "${TAR_CMD[*]}"
    exit 0
fi

command -v "$CURL" >/dev/null 2>&1 || die "$CURL is not installed"
command -v tar >/dev/null 2>&1 || die "tar is not installed"

BODY="$(mktemp)"
trap 'rm -f "$BODY"' EXIT

printf '==> asking Portainer (%s) to build %s from target %s\n' "$BASE" "$TAG" "$TARGET"
# The response goes to stdout so it can be rendered as it arrives - a CUDA build
# is a long silence otherwise - and `tee` keeps a copy so the error object can be
# found after the fact. The HTTP status is appended by -w as the final line.
if ! "${TAR_CMD[@]}" | "$CURL" "${CURL_ARGS[@]}" -w '\n%{http_code}' | tee "$BODY" | render_stream; then
    die "the build request to Portainer failed (is PORTAINER_URL reachable, and the token valid?)"
fi

STATUS="$(tail -n 1 "$BODY" | tr -d '\r' | tr -dc '0-9')"
printf '\n'
if [[ -n "$STATUS" ]] && ((STATUS >= 400)); then
    die "Portainer rejected the build request with HTTP ${STATUS}"
fi
if grep -q '"error"' "$BODY" 2>/dev/null; then
    die "the build failed; the error is in the output above"
fi
printf '==> %s built on the Portainer host\n' "$TAG"

if [[ "$VERIFY" == "1" ]]; then
    # The image list is the authoritative answer to "is it there?" - the build
    # stream ending successfully is not, if the tag was not what we asked for.
    CHECK_ARGS=(-sS -H "X-API-Key: ${PORTAINER_API_TOKEN}" "${BASE}/images/json")
    [[ "$INSECURE" == "1" ]] && CHECK_ARGS+=(-k)
    if "$CURL" "${CHECK_ARGS[@]}" | grep -q -- "$TAG"; then
        printf '==> verified: %s exists on the Portainer host\n' "$TAG"
    else
        die "the build finished but $TAG is not in the image list on the Portainer host"
    fi
fi

cat <<EOF

Deploy it with a stack that names this image (docs/portainer.md), then trigger the
redeploy without a pull:

    PORTAINER_WEBHOOK_URL=https://portainer.internal:9443/api/stacks/webhooks/<id> \\
      scripts/portainer_deploy.sh
EOF

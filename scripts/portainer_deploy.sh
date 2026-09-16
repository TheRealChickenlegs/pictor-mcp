#!/usr/bin/env bash
#
# Redeploy a pictor-mcp stack in Portainer, without pulling an image.
#
#   PORTAINER_WEBHOOK_URL=https://portainer.internal:9443/api/stacks/webhooks/<id> \
#     scripts/portainer_deploy.sh
#
# Stack webhooks are a Portainer Business feature. The interesting part is
# `pullimage=false`: by default a webhook redeploy pulls the stack's images, and
# for an image that was just built on this host that pull would either fail (the
# local tag is in no registry) or drag gigabytes back down the wire. Portainer's
# documented way to prevent it is exactly this parameter.
#
# This script is the "and now deploy it" half of a local build. See
# docs/portainer.md for the stack itself and the git-push wiring.
#
# Environment:
#   PORTAINER_WEBHOOK_URL  required, the stack webhook URL from Portainer
#   PORTAINER_INSECURE=1   skip TLS verification (self-signed certificate)
#   CURL                   curl binary to use (default: curl)

set -euo pipefail

CURL="${CURL:-curl}"
INSECURE="${PORTAINER_INSECURE:-0}"
ALLOW_PULL=0
STACK_TAG=""
PRINT_ONLY=0

usage() {
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
    printf 'portainer_deploy.sh: %s\n' "$*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            [[ $# -ge 2 ]] || die "--tag needs a value"
            STACK_TAG="$2"
            shift 2
            ;;
        --pull)
            # Opt back in to pulling, for the case where the image really does
            # come from a registry.
            ALLOW_PULL=1
            shift
            ;;
        --insecure)
            INSECURE=1
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

: "${PORTAINER_WEBHOOK_URL:?set PORTAINER_WEBHOOK_URL to the stack webhook URL from Portainer}"
case "$PORTAINER_WEBHOOK_URL" in
    */api/stacks/webhooks/*) : ;;
    *) die "that does not look like a stack webhook URL (expected .../api/stacks/webhooks/<id>)" ;;
esac

# The URL may already carry a query; appending with ? unconditionally would
# silently produce a second query string and drop both parameters.
SEPARATOR="?"
case "$PORTAINER_WEBHOOK_URL" in
    *\?*) SEPARATOR="&" ;;
esac

PARAMS=""
if [[ "$ALLOW_PULL" == "0" ]]; then
    PARAMS="pullimage=false"
fi
if [[ -n "$STACK_TAG" ]]; then
    PARAMS="${PARAMS:+${PARAMS}&}tag=${STACK_TAG}"
fi
URL="$PORTAINER_WEBHOOK_URL"
[[ -n "$PARAMS" ]] && URL="${URL}${SEPARATOR}${PARAMS}"

CURL_ARGS=(-sS -X POST "$URL")
[[ "$INSECURE" == "1" ]] && CURL_ARGS+=(-k)

if [[ "$PRINT_ONLY" == "1" ]]; then
    printf 'curl %s\n' "${CURL_ARGS[*]}"
    exit 0
fi

command -v "$CURL" >/dev/null 2>&1 || die "$CURL is not installed"

STATUS="$("$CURL" "${CURL_ARGS[@]}" -o /dev/null -w '%{http_code}')" ||
    die "the webhook call failed (is PORTAINER_WEBHOOK_URL reachable?)"

if [[ "$STATUS" =~ ^2 ]]; then
    printf '==> redeploy requested (HTTP %s)\n' "$STATUS"
    cat <<'EOF'

Portainer answers a stack webhook before the deployment finishes, so a 2xx means
"accepted", not "running". Follow it in the stack's log, or check the container:

    docker ps --filter name=pictor-mcp
EOF
else
    die "Portainer answered HTTP ${STATUS}"
fi

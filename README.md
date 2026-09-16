# pictor-mcp

A secure [Model Context Protocol](https://modelcontextprotocol.io) server for image
operations, packaged as a hardened Docker image.

Point any MCP client at it and the model can inspect, convert, resize, compress,
crop, rotate, watermark, background-remove and batch-process images — and verify
its own edits afterwards.

```bash
mkdir -p input output
cp .env.example .env            # set PICTOR_AUTH_TOKEN
docker compose up -d            # pulls the published image; nothing is built
```

The server is then at `http://127.0.0.1:8077/mcp`.

---

## Contents

- [Why this one](#why-this-one)
- [Quick start](#quick-start)
  - [File ownership](#file-ownership)
  - [Which compose file](#which-compose-file)
  - [Pinning a version](#pinning-a-version)
- [Client setup](#client-setup)
  - [DeepSeek Harness (DSH)](#deepseek-harness-dsh)
  - [OpenCode](#opencode)
  - [Hermes agent](#hermes-agent)
  - [Open WebUI](#open-webui)
  - [Any other stdio client](#any-other-stdio-client)
- [Tools](#tools)
- [How results come back](#how-results-come-back)
- [GPU acceleration](#gpu-acceleration)
- [Security](#security)
- [Configuration](#configuration)
- [Running without Docker](#running-without-docker)
- [Container images](#container-images)
- [Development](#development)

---

## Why this one

**It is a file primitive, so it is treated like one.** An image server that will
read any path an agent names is arbitrary-file-read; one that writes any path is
arbitrary-file-write. Reads are confined to configured roots, writes to a single
output root, symlink escapes and traversal are refused, and the error never
reveals whether a forbidden path exists.

**It survives hostile images.** Decompression bombs, over-long axes, animation
amplification and truncated files are rejected before pixel data is allocated.

**It returns what every client can use.** Each result carries a self-contained
text summary, machine-readable `structuredContent`, an optional inline image for
vision models, and a resource link — so the same call works in a terminal, a
chat UI, or an agent loop.

**It speaks every MCP revision.** The MCP Python SDK negotiates per connection:
modern stateless `2026-07-28` requests and legacy `initialize`-handshake clients
are served from the same endpoint with no compatibility flag.

**It composes.** `image_transform` runs an ordered operation list in one call, so
an agent does not pay a round trip and a base64 transfer per step.

---

## Quick start

The compose files pull a prebuilt image from GHCR, so there is nothing to
compile:

```bash
git clone https://github.com/TheRealChickenlegs/pictor-mcp.git
cd pictor-mcp

mkdir -p input output
cp .env.example .env
openssl rand -hex 32            # paste into PICTOR_AUTH_TOKEN in .env

# If your uid or gid is not 1000, put your own in .env (see below)
docker compose up -d
```

Verify it:

```bash
docker pull ghcr.io/therealchickenlegs/pictor-mcp:latest   # confirm the tag exists
curl -s http://127.0.0.1:8077/healthz                      # {"status":"ok"}
docker compose logs -f pictor-mcp
docker compose exec pictor-mcp python -m pictor_mcp --check # resolved config, secrets redacted
```

Put images in `./input`, and the server writes results to `./output`.

The default port mapping is `127.0.0.1:8077:8077`, so **nothing off-host can
reach it**. To expose it on your LAN, see
[Exposing beyond localhost](#exposing-beyond-localhost).

### File ownership

The container runs as `PUID:PGID`, set in `.env` and defaulting to `1000:1000`.
Point them at your own identity and everything the server writes to `./output`
belongs to you, and `./input` is readable without loosening anything:

```bash
id -u    # e.g. 1000  ->  PUID
id -g    # e.g. 1000  ->  PGID
```

That is why the quick start above needs no `chown`. If you would rather not
change the container's identity, the other direction works too:

```bash
sudo chown -R 1000:1000 input output    # match the container's default
```

Getting this wrong is the one setup error you are likely to hit, and it is
deliberately loud rather than silent — the server refuses to start rather than
booting and failing every call:

```
configuration error: output root /data/output is not writable by uid 1000,
gid 1000: Permission denied. In Docker the container user must match the owner
of the mounted host directory ...
```

Do **not** set `PUID` or `PGID` to `0`. Running as root would negate
`cap_drop`, `no-new-privileges` and the read-only root filesystem, and it is
unnecessary: `/data/output` inside the image is writable by any uid precisely so
that this choice is free.

### Which compose file

| File | What it does |
|---|---|
| `docker-compose.yml` | CPU image. The default; `docker compose up -d`. |
| `docker-compose.gpu.yml` | Overlay: switches to the CUDA image and passes the GPU through. |
| `docker-compose.ml.yml` | Overlay: the ML image (CUDA + background removal, u2net baked in). |
| `docker-compose.build.yml` | Overlay: build from this checkout instead of pulling. |

Overlays are combined with repeated `-f`, and they only add — the hardening,
volumes and environment all come from the base file, so a variant cannot drift
from it:

```bash
# GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# ML (background removal); do not combine with the GPU overlay, they use the
# same host port and both request the GPU
docker compose -f docker-compose.yml -f docker-compose.ml.yml up -d
```

### Pinning a version

`latest` follows the default branch. For a reproducible deployment, set exact
tags in `.env` — none of these are read by the server, they are Compose knobs:

```bash
IMAGE_TAG=1.0.0
IMAGE_TAG_GPU=1.0.0-gpu
IMAGE_TAG_ML=1.0.0-ml
```

Every image also carries an immutable `sha-<short>` tag, which is the one to
pin if you want to be certain nothing moves.

---

## Client setup

Every client below needs the same three things: the URL
`http://<host>:8077/mcp`, a bearer token, and nothing else.

### DeepSeek Harness (DSH)

Add to your DSH plugin configuration. DSH exposes the tools as
`mcp__pictor__image_resize` and so on.

```yaml
- id: mcp-pictor
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: pictor
    transport: streamable-http
    url: http://127.0.0.1:8077/mcp
    headers:
      Authorization: !!js '`Bearer ${process.env.PICTOR_AUTH_TOKEN}`'
    toolCallTimeoutMs: 180000
```

DSH renders the inline `image` content blocks, so a vision model can see the
result directly. Raise `toolCallTimeoutMs` above the default 60 s if you process
very large images.

### OpenCode

In `~/.config/opencode/opencode.json` (or a project `opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "pictor": {
      "type": "remote",
      "url": "http://127.0.0.1:8077/mcp",
      "enabled": true,
      "headers": {
        "Authorization": "Bearer ${PICTOR_AUTH_TOKEN}"
      }
    }
  }
}
```

For a local (stdio) server instead of Docker:

```json
{
  "mcp": {
    "pictor": {
      "type": "local",
      "command": ["python", "-m", "pictor_mcp"],
      "environment": {
        "PICTOR_TRANSPORT": "stdio",
        "PICTOR_INPUT_ROOTS": "/home/you/images",
        "PICTOR_OUTPUT_ROOT": "/home/you/images/out"
      },
      "enabled": true
    }
  }
}
```

### Hermes agent

Hermes keeps MCP servers in `~/.hermes/config.yaml` under `mcp_servers`, or you
can use the CLI: `hermes mcp add`.

```yaml
mcp_servers:
  pictor:
    url: "http://127.0.0.1:8077/mcp"
    headers:
      Authorization: "Bearer YOUR_TOKEN_HERE"
    timeout: 180
    connect_timeout: 10
```

Notes:

- Omit `transport` for Streamable HTTP (the default). Add `transport: sse` only
  if you deliberately started the deprecated SSE transport.
- If Hermes' pre-flight content-type probe rejects the endpoint, set
  `skip_preflight: true` — the probe expects a content type that a valid
  Streamable HTTP endpoint does not have to return.
- For a stdio server, replace `url`/`headers` with `command: "python"` and
  `args: ["-m", "pictor_mcp"]`, plus an `env:` block.

### Open WebUI

Open WebUI talks to MCP tool servers from its backend (server-side), so no
browser-origin configuration is needed.

1. **Admin Settings → Tools → Add Connection / MCP Servers**, add a
   **Streamable HTTP** entry with URL `http://pictor-mcp:8077/mcp` (container
   name, if Open WebUI shares a Docker network) or
   `http://host.docker.internal:8077/mcp` from a container to a host port.
2. Set the auth header to `Authorization: Bearer <your token>`.
3. Make the tools available to a model, then ask it to resize an image.

Two Open WebUI specifics worth knowing:

- **Images render best via URL.** Turn on
  [`PICTOR_SERVE_OUTPUTS`](#generated-file-urls) so results include a markdown
  image link Open WebUI can display. Otherwise the model sees only the text summary.
- **If tool discovery fails**, try `PICTOR_STATELESS_HTTP=true` (the default) and
  `PICTOR_JSON_RESPONSE=true`. Some Open WebUI versions handle plain JSON
  responses better than SSE streams.

### Any other stdio client

The server runs as a normal stdio MCP server:

```bash
pip install .
PICTOR_INPUT_ROOTS=/path/to/images PICTOR_OUTPUT_ROOT=/path/to/out python -m pictor_mcp
```

Or with Docker:

```bash
docker run -i --rm \
  --user "$(id -u):$(id -g)" \
  -v /path/to/images:/data/input:ro \
  -v /path/to/out:/data/output \
  -e PICTOR_TRANSPORT=stdio \
  ghcr.io/therealchickenlegs/pictor-mcp:latest
```

`--user` is what `docker compose` sets from `PUID`/`PGID`; without it the
container uses the image's own uid, and `/path/to/out` would need to be writable
by that account instead. See [File ownership](#file-ownership).

---

## Tools

14 tools. `image_capabilities` reports exactly which are usable in your
deployment, so an agent can check rather than guess.

### Inspect

| Tool | Purpose |
|---|---|
| `image_capabilities` | Formats, operations, limits, security posture, GPU status. |
| `image_info` | Dimensions, format, mode, frames, EXIF, perceptual hashes, dominant colours. |
| `image_compare` | SSIM, RMSE, PSNR, changed-pixel ratio and hash distance between two images. |

### Transform

| Tool | Purpose |
|---|---|
| `image_convert` | Between JPEG, PNG, WebP, AVIF, TIFF, GIF, BMP, ICO, JPEG 2000, QOI, PPM. |
| `image_resize` | By width, height, percentage, or into a box. Six fit modes, six filters. |
| `image_compress` | By quality, or search for the best quality under a target size. |
| `image_crop` | Pixel box, aspect ratio with gravity, or auto-trim a border. |
| `image_rotate` | Any angle, mirror, and EXIF auto-orient. |
| `image_thumbnail` | Exact-size thumbnail, cropped by saliency rather than blindly centred. |
| `image_watermark` | Text or image, positioned, rotated, tiled, with opacity. |
| `image_background_remove` | Colour keying (offline, instant) or an ML model. |
| `image_optimize_web` | A width ladder of web variants plus a placeholder and a `srcset`. |

### Compose and batch

| Tool | Purpose |
|---|---|
| `image_transform` | An ordered operation pipeline in one call. |
| `image_batch` | The same pipeline across many files, with per-file error isolation. |

### `image_transform`

Prefer this when you need more than one step. Operations are validated before
any pixels are touched:

```json
{
  "path": "photo.jpg",
  "operations": [
    {"op": "auto_orient"},
    {"op": "resize", "width": 1200, "fit": "cover", "height": 630},
    {"op": "watermark_text", "text": "DRAFT", "opacity": 0.35, "tile": true},
    {"op": "sharpen", "amount": 1.1}
  ],
  "output_format": "webp",
  "quality": 82
}
```

Available operations: `auto_orient`, `resize`, `crop`, `rotate`, `flip`,
`sharpen`, `blur`, `smart_crop`, `watermark_text`, `watermark_image`,
`background_remove`, `feather`.

Fit modes: `contain` (fit inside), `cover` (fill and crop), `fill` (stretch),
`inside` (contain, never enlarge), `outside` (cover, no crop), `pad` (contain
then pad to the exact box).

### Image inputs

Every tool accepts **exactly one** of:

- `path` — relative to an allowed root, or absolute inside one. Reads also work
  from the output root, so you can read back what the server just produced.
- `base64_data` — inline bytes, optionally a `data:` URI.
- `url` — HTTP(S), **disabled by default** and SSRF-guarded when enabled.

---

## How results come back

One call, four representations of the same artefact — pick whichever your client
understands:

```jsonc
// 1. text — always present, self-contained
"Image resize completed.\nInput: 4000x3000 JPEG 3.1 MB\nOutput: 1200x900 WEBP 142.3 KB -> resized/photo-w1200.webp\nSize: 3.1 MB -> 142.3 KB (95.4% smaller)",

// 2. image — inline, for vision models (when return_image is true and it fits)
{"type": "image", "data": "<base64>", "mimeType": "image/webp"},

// 3. resource_link — for clients that resolve MCP resources
{"type": "resource_link", "uri": "pictor://outputs/resized/photo-w1200.webp", ...},

// 4. structuredContent — for programmatic callers
{
  "ok": true,
  "operation": "image_resize",
  "outputs": [{
    "name": "photo-w1200.webp", "path": "resized/photo-w1200.webp",
    "mimeType": "image/webp", "format": "webp", "byteSize": 145715,
    "width": 1200, "height": 900, "sha256": "…", "url": "http://…/files/…"
  }],
  "sizeChange": {"inputBytes": 3251840, "outputBytes": 145715, "savedPercent": 95.52}
}
```

Base64 never appears twice: if the image is inlined, it is not also in the JSON.

**Errors are readable and machine-actionable.** A refused path returns
`is_error: true`, a plain message (`path is outside the configured input roots`),
and a stable code in `structuredContent.error.code`. Messages never contain host
paths or library internals, so they are safe to show a model.

---

## GPU acceleration

The CPU image is the default and needs no GPU. If you have an NVIDIA card,
switch to the CUDA image with an overlay; acceleration is then detected
automatically.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and a driver visible to `nvidia-smi`. Sanity-check the host before blaming the
image:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

The GPU tags are `:gpu` and `:<version>-gpu`; the overlay selects them for you,
and `IMAGE_TAG_GPU` pins an exact one.

**The CUDA line must cover your card's architecture.** The image installs a
CUDA 12.8 build (`cu128`), which carries kernels for Turing through Blackwell
(RTX 50-series). A card newer than the build will load, appear available, and
then fail every kernel launch. The server detects this at startup, disables the
GPU, and says so in plain terms:

```
GPU acceleration unavailable: RTX 5060 Ti (torch ..., cuda ...) is sm_120, which
this PyTorch build has no kernels for; it supports sm_50, ..., sm_90. This build
predates that architecture. Rebuild with a CUDA 12.8 or newer wheel index ...
```

The same text appears in `image_capabilities`, so an agent can report it. To
change it, override the index at build time — and note the coupling, since an
index only helps if it still publishes wheels for the base image's Python:

```yaml
build:
  args:
    TORCH_INDEX_URL: "https://download.pytorch.org/whl/cu129"
```

Check a pairing before committing to it:

```bash
curl -s https://download.pytorch.org/whl/cu128/torch/ | grep -o 'cp3[0-9]*' | sort -u
```

What actually speeds up, and what does not:

- **Resampling** (resize) runs on the GPU via `torch.nn.functional.interpolate`.
  It is only used above `PICTOR_GPU_MIN_PIXELS` (default 4 MP), because below
  that the host↔device copy costs more than the resize.
- **Encoding stays on the CPU.** libjpeg-turbo and libwebp have no CUDA path in
  Pillow and are already fast.
- **Lanczos is approximated** by bicubic with antialiasing, since torch has no
  Lanczos kernel. The result notes say so when it happens.
- **Background removal** with `method: "ml"` uses `onnxruntime-gpu` in the `ml`
  image.

The GPU is an accelerator, never a requirement. On startup the server resizes a
test image on both paths and compares them; if they disagree beyond tolerance the
backend disables itself and logs why. A missing driver, an out-of-memory card or
an unsupported pixel format degrades throughput, never correctness.

```bash
# Ask the server what it is actually using
# (call the image_capabilities tool, or read the startup log line)
docker compose exec pictor-mcp python -m pictor_mcp --check | grep -i accel
```

### The `ml` image

Adds rembg + `onnxruntime-gpu` on top of the CUDA image, with the u2net
segmentation weights baked in at build time. That baking is the point: the
container needs no network at runtime, which is what makes it usable on an
isolated internal network.

```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml up -d
```

It is substantially larger (roughly a gigabyte of model and ONNX runtime on top
of the CUDA wheels), so use it only if you actually need ML background removal.
The CPU image can already remove a flat background exactly and instantly with
`method: "color"`, which covers the usual product or logo shot.

Only `u2net` ships inside the image, so only that model works offline. Any other
`PICTOR_BG_MODEL` is downloaded on first use and needs a writable or pre-seeded
directory at `U2NET_HOME` — see the commented volume in `docker-compose.ml.yml`.

Do not combine the `ml` and `gpu` overlays: they publish the same host port and
both request the GPU.

### Building from source

For a local modification, add the build overlay. Note it also needs `target:`
changed in that file to build the CUDA or ML stage, otherwise Compose builds the
CPU stage and tags the result with the CUDA tag:

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build
```

---

## Security

Read [SECURITY.md](SECURITY.md) for the full threat model. The short version:

| Risk | Control |
|---|---|
| Arbitrary file read | All reads confined to `PICTOR_INPUT_ROOTS` + the output root. Traversal, symlink escapes, absolute paths and special files (FIFOs, devices) refused. |
| Arbitrary file write | All writes confined to `PICTOR_OUTPUT_ROOT`, written atomically with `O_NOFOLLOW`. |
| TOCTOU / symlink swap | Files are opened with `O_NOFOLLOW` and re-verified through `/proc/self/fd` after opening. |
| Decompression bombs | Pixel, per-axis, frame and file-size ceilings enforced **before** allocation. |
| SSRF via URL input | Off by default. When on: scheme/port/host allow-lists, all resolved addresses must be globally routable, connection pinned to the validated IP, per-hop redirect validation, streaming size cap. |
| DNS rebinding | Host and Origin headers validated on **every** route, including health and file serving. |
| Unauthenticated access | Optional bearer token, compared in constant time. Fail-closed configuration: serving files without a credential refuses to start. |
| Metadata leakage | EXIF, GPS and ICC stripped by default. |
| Stored XSS via output files | Served content types are allow-listed to images; anything else is an opaque attachment, every response carries a sandbox CSP, and `output_name` cannot choose its own extension. |
| Unbounded intermediate allocation | `cover`/`outside` resizes and saliency thumbnails bound the *scaled* bitmap, not just the result; `image_compare` processes in strips so memory is independent of image size. |
| Container breakout | Non-root runtime (`PUID:PGID`, default `1000:1000`, never `0`), read-only root filesystem, all capabilities dropped, `no-new-privileges`, PID and memory limits. |
| Resource exhaustion | Concurrency cap, per-call timeouts, request body cap, batch limits. |
| Secret leakage in logs | Startup secrets are redacted; stdout is never used for logging (it is the stdio protocol channel). |

### Exposing beyond localhost

The default port mapping binds to loopback, so nothing off-host can reach the
server. Making it reachable is a three-part change, and doing only the first
part is the mistake worth avoiding.

**1. Bind the port.** The published port is controlled by `BIND_ADDRESS` in
`.env`, which is easier to switch back than editing the mapping:

```bash
# .env - reachable from anything that can route to this host
BIND_ADDRESS=0.0.0.0
```

The `ports:` block in `docker-compose.yml` also lists the equivalent mappings as
commented alternatives, including binding one specific interface
(`192.168.1.10:8077:8077`), which is the safest option when the host has several
addresses (VPN, `docker0`, a second NIC).

**2. Set the allow-lists to match**, or browser-originated requests are refused
by design. The defaults name only loopback and the compose service name:

```bash
PICTOR_ALLOWED_HOSTS=192.168.1.10:8077,pictor.internal:8077
PICTOR_ALLOWED_ORIGINS=http://192.168.1.10:8077
```

Non-browser MCP clients (DSH, OpenCode, Hermes, Open WebUI's backend) send no
`Origin` header and are unaffected either way, which is why the loopback default
is safe. A browser-based client is the case that needs the allow-list.

Both lists accept a small pattern grammar: a bare `host` or `host:port` matches
exactly, `host:*` matches that host on any port, `*.example.com` matches any
subdomain on any port (never the apex itself), and `*` matches everything — only
sane behind a hardened proxy. `PICTOR_ALLOWED_ORIGINS` takes the same forms with
an optional scheme, e.g. `https://*.example.com`. These patterns are interpreted
by this server; the MCP SDK's weaker built-in Host check is disabled so there is
only one answer to "is this request allowed?".

**3. Always set `PICTOR_AUTH_TOKEN`** once anything but you can reach the port.
Anyone who can reach an unauthenticated instance has the full tool surface, which
reads and writes files.

Better still, terminate TLS in a reverse proxy and set `PICTOR_ENABLE_HSTS=true`;
the server speaks plain HTTP by design, so put Caddy/nginx/Traefik in front if
the network is not trusted. When the proxy makes the external host or port differ
from the bind address, also set `PICTOR_PUBLIC_BASE_URL` so generated file links
point somewhere useful.

### Connecting from another container

This is not the same as exposing the port, and the port mapping is irrelevant to
it: containers on one Docker network reach each other by service name on the
container's own port, never through the host's published port. A client that
calls `http://pictor-mcp:8077/mcp` sends `Host: pictor-mcp:8077`, so the
`PICTOR_ALLOWED_HOSTS` default includes `pictor-mcp:*` and this works with no
extra configuration.

Renaming the service in `docker-compose.yml` renames the Host header too, so
update the allow-list to match. If a client is refused with

```
rejected request: host header 'X' is not allowed
```

then `X` is the value to add to `PICTOR_ALLOWED_HOSTS` — an aggregator that
reaches the server through a reverse proxy, or by a LAN name, sends that name
instead. The rejection is logged before authentication is checked, so a Host
rejection is not an indication that the token is wrong; fix the Host list first,
then confirm the token.

### Generated file URLs

With `PICTOR_SERVE_OUTPUTS=true`, results include a URL so a web UI can render
the image. Those URLs are **not** bearer-authenticated — they carry an HMAC
signature with an expiry, scoped to one file, which is what lets a browser
`<img>` tag work without the API token.

- The server **refuses to start** with `PICTOR_SERVE_OUTPUTS=true` unless
  `PICTOR_AUTH_TOKEN` or `PICTOR_URL_SECRET` is set, so this can never silently
  become an open directory of everything the server has produced.
- Set `PICTOR_PUBLIC_BASE_URL` when the externally visible host/port differs from
  the bind address (reverse proxy, different published port), or the generated
  links will point at `127.0.0.1:<PICTOR_PORT>`.
- `PICTOR_URL_TTL_SECONDS` (default 3600) bounds how long a leaked link works.

---

## Configuration

Every setting is an environment variable prefixed `PICTOR_`. Full annotated list
with defaults: [`.env.example`](.env.example). Validate a deployment without
starting it:

```bash
docker compose exec pictor-mcp python -m pictor_mcp --check
```

The most consequential ones:

| Variable | Default | Notes |
|---|---|---|
| `PICTOR_TRANSPORT` | `stdio` (`streamable-http` in Docker) | `stdio`, `streamable-http` or `sse`. |
| `PICTOR_HOST` / `PICTOR_PORT` | `127.0.0.1` / `8077` | Loopback by default outside Docker. |
| `PICTOR_INPUT_ROOTS` | `/data` | Read roots, comma-separated, absolute. |
| `PICTOR_OUTPUT_ROOT` | `/data/output` | The only writable directory. Must not overlap an input root. |
| `PICTOR_AUTH_TOKEN` | _(empty)_ | Required for HTTP once reachable off-host. Minimum 16 characters. |
| `PICTOR_ALLOW_NET_FETCH` | `false` | Enables URL inputs, with the SSRF guard. |
| `PICTOR_SERVE_OUTPUTS` | `false` | Signed URLs for generated files. |
| `PICTOR_PUBLIC_BASE_URL` | _(empty)_ | External base URL for generated links. |
| `PICTOR_STATELESS_HTTP` | `true` | Best client compatibility. |
| `PICTOR_MAX_PIXELS` | `64000000` | Per-frame decompression-bomb ceiling. |
| `PICTOR_MAX_ANIMATION_PIXELS` | `128000000` | Ceiling on width × height × frames. |
| `PICTOR_OP_TIMEOUT_SECONDS` | `120` | Cooperative budget, checked between pipeline steps and quality probes. |
| `PICTOR_HTTP_ACCESS_LOG` | `false` | uvicorn access log; off so signed URLs are not written to logs. |
| `PICTOR_MAX_CONCURRENCY` | `4` | Concurrent operations; size with `mem_limit`. |
| `PICTOR_STRIP_METADATA` | `true` | Strip EXIF/GPS/ICC from outputs. |
| `PICTOR_ALLOWED_HOSTS` | loopback + `pictor-mcp:*` in Docker | Host headers accepted. Add a name if a client is refused; see [Connecting from another container](#connecting-from-another-container). |
| `PICTOR_GPU` | `off` (`auto` in the GPU image) | `auto`, `off` or `torch`. |

Invalid values make the server **refuse to start** rather than fall back to a
less safe default.

Separately, section 0 of `.env.example` holds the **Compose-only** settings.
They have no `PICTOR_` prefix precisely so they cannot be mistaken for server
options:

| Variable | Default | Notes |
|---|---|---|
| `IMAGE_REPO` | `ghcr.io/therealchickenlegs/pictor-mcp` | Registry path, no tag. |
| `IMAGE_TAG` | `latest` | CPU image tag. Pin e.g. `1.0.0`. |
| `IMAGE_TAG_GPU` | `gpu` | CUDA image tag, e.g. `1.0.0-gpu`. |
| `IMAGE_TAG_ML` | `ml` | ML image tag, e.g. `1.0.0-ml`. |
| `BIND_ADDRESS` | `127.0.0.1` | Host interface the port binds to. `0.0.0.0` for LAN. |
| `PUID` / `PGID` | `1000` / `1000` | uid:gid the container runs as. Set to `id -u` / `id -g` so `./output` files are yours. Never `0`. |

---

## Running without Docker

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install .                 # or ".[gpu]", ".[bg]", ".[all]"

PICTOR_TRANSPORT=stdio \
PICTOR_INPUT_ROOTS="$HOME/Pictures" \
PICTOR_OUTPUT_ROOT="$HOME/Pictures/out" \
python -m pictor_mcp
```

For an HTTP server natively, set `PICTOR_TRANSPORT=streamable-http` and
`PICTOR_HOST=127.0.0.1` (a non-loopback host with no explicit
`PICTOR_ALLOWED_HOSTS` accepts any Host header — pair the two).

---

## Container images

Three variants are published to the GitHub Container Registry on every push to
the default branch and every `v*` tag:

| Tag | Contents | Platforms |
|---|---|---|
| `ghcr.io/therealchickenlegs/pictor-mcp:latest` | CPU, ~180 MB | `linux/amd64`, `linux/arm64` |
| `ghcr.io/therealchickenlegs/pictor-mcp:gpu` | CPU + PyTorch CUDA wheels | `linux/amd64` |
| `ghcr.io/therealchickenlegs/pictor-mcp:ml` | GPU + rembg + `onnxruntime-gpu`, u2net baked in | `linux/amd64` |

Version tags are added alongside (`1.0.0`, `1.0`, `1.0.0-gpu`, …), plus an
immutable `sha-<short>` tag per commit. The CUDA images are amd64-only because
PyTorch does not publish `linux/arm64` wheels for the CUDA index they install
from. Images carry a signed build-provenance attestation and an SBOM.

The compose files already point at these tags, so `docker compose up -d` pulls
rather than builds. Override `IMAGE_REPO` if you mirror them elsewhere, and
`IMAGE_TAG*` to pin versions.

A published image reports the version baked into it, so you can always tell what
you are running:

```bash
docker run --rm ghcr.io/therealchickenlegs/pictor-mcp:latest python -m pictor_mcp --version
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ruff check src tests
ruff format src tests
pytest -q
```

The suite (≈400 tests) covers path confinement and symlink escapes, SSRF
classification of internal address ranges, decompression bombs, format
allow-listing, encoder fallbacks, pipeline semantics, and both transports
end-to-end — including a live HTTP server for the auth, DNS-rebinding and signed
URL behaviour.

### Checks

Pull requests run five jobs, all of which must pass:

| Job | What it enforces |
|---|---|
| **Lint, format and syntax** | `ruff check` (which parses every file, so syntax errors surface first), `ruff format --check`, and `compileall`. |
| **Lint the workflows** | `actionlint`, whose image bundles `shellcheck` and `pyflakes`, so the shell and Python embedded in the workflows are checked too. |
| **Tests** | The full suite on Python 3.10, 3.11, 3.12 and 3.13 — the declared floor, the version the image ships, and the newest the SDK supports. |
| **Build and install the distribution** | Builds the sdist and wheel, installs the wheel into a clean venv, runs the console script, and smoke-tests the installed server over stdio. |
| **Deployment files** | Parses `docker-compose.yml`, the GPU overlay and `.env.example` and pushes them through the real config parser, so a renamed setting or a lost hardening flag fails the build. |

`CodeQL` runs `security-extended` analysis on pushes, pull requests and weekly.
`Dependabot` keeps the actions, Python dependencies and base image current.

Layout:

```
src/pictor_mcp/
├── config.py          environment parsing, fail-loud
├── server.py          transports, middleware, wiring
├── outputs.py         result envelope
├── models.py          public result schema
├── errors.py          error taxonomy with stable codes
├── security/          path jail, limits, SSRF guard, auth, concurrency
├── imaging/           formats, loader, ops, encode, pipeline, analysis
├── backends/          pluggable CPU/CUDA resampling
└── tools/             the 14 MCP tools
```

## License

MIT. See [LICENSE](LICENSE).

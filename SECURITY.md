# Security model

pictor-mcp is a **file-read and file-write primitive with an image API on top**.
That framing drives every decision below. "Resize this PNG" is a friendly
description of "open a path an untrusted model named, hand you back the bytes,
and write a file somewhere".

This document describes what is defended, what is explicitly not, and how to
deploy it safely.

---

## Reporting a vulnerability

Open a private security advisory on the repository, or contact the maintainer
directly. Please include a reproduction. Do not open a public issue for anything
exploitable.

---

## Threat model

**Trusted:** the operator, the host, the network the container is deployed on,
and the container runtime.

**Untrusted:** everything the model sends — paths, base64 payloads, URLs, image
contents, operation parameters — and any image file it points at. A model can be
prompt-injected by an image's metadata, a web page it fetched, or any other tool
it has; assume it may be actively adversarial.

**In scope:** filesystem escape, resource exhaustion, SSRF into the host or LAN,
authentication bypass, information disclosure, container escape, and
supply-chain risk in the build.

**Out of scope:** a compromised host or runtime, an attacker who already has the
bearer token *and* network access, and denial of service by an attacker who can
simply stop the container.

---

## Controls

### 1. Filesystem confinement

Every read and write goes through a single `PathJail`.

| Technique | Control |
|---|---|
| `../../etc/passwd` | Paths are normalised (`os.path.normpath`) and then checked for containment against a configured root. |
| Absolute paths outside the roots | Refused; containment is checked after resolution, not before. |
| Symlink pointing outside a root | Resolution follows links, so the escape is visible to the containment check. |
| Symlinked parent directory on write | Every existing ancestor is resolved and validated *before* the write. |
| TOCTOU: swapping a file for a symlink after validation | Files are opened with `O_NOFOLLOW`; the open descriptor's real path is then re-verified via `/proc/self/fd`. |
| FIFOs, devices, `/dev/zero` | Opened `O_NONBLOCK`, then rejected unless the descriptor is a regular file. Without `O_NONBLOCK` a FIFO would hang the worker thread forever *before* the `S_ISREG` check could run. |
| Filesystem oracle ("does `/etc/shadow` exist?") | A confinement failure returns "path is outside the configured input roots" — never whether the target exists, and never a host path. |
| NUL bytes, control characters, 4096+ byte paths, backslash separators | Rejected up front. |
| Unicode look-alike paths | NFC-normalised before checking. |
| Output filename injection | Derived filenames are sanitised (`safe_filename`): separators, control characters, leading dots and reserved stems removed. |

Reads are allowed from the configured input roots **and** the output root — the
latter so an agent can read back what the server produced (resize, then compare).
That is server-owned space, so the trust boundary is unchanged. Writes are
restricted to the output root alone.

### 2. Resource limits

Decoding happens only after the declared geometry is checked, so a bomb is
rejected before any pixel buffer is allocated.

| Limit | Default | Guards against |
|---|---|---|
| `PICTOR_MAX_FILE_BYTES` | 64 MiB | Oversized uploads; enforced while streaming, not from a header. |
| `PICTOR_MAX_PIXELS` | 64 MP | Decompression bombs. Also set as Pillow's `MAX_IMAGE_PIXELS` as a backstop. |
| `PICTOR_MAX_DIMENSION` | 24000 | Over-long single axes, which some C decoders handle worse than large areas. |
| `PICTOR_MAX_FRAMES` | 120 | Animation amplification; the frame table is walked with the ceiling enforced as it goes. |
| `PICTOR_MAX_ANIMATION_PIXELS` | 128 MP | The *product* of frames and pixels. Per-frame and frame-count limits do not bound it, and every frame is held as a full bitmap before re-encoding. |
| `PICTOR_OP_TIMEOUT_SECONDS` | 120 | Cooperative wall-clock budget, checked between pipeline steps and between quality-search probes. A single Pillow call cannot be interrupted from Python, so this bounds *sequences* of work, not one call. |

Intermediate allocations are bounded too, not just final ones. `cover` and
`outside` fit modes scale until the target box is covered, so a `24000x1`
target on a square source produces a 24000x24000 intermediate (~2.3 GB as RGBA)
before cropping back down. The check is therefore applied to the scaled bitmap
and to the saliency thumbnail's scaled copy, and `image_optimize_web` (which
encodes directly rather than through the pipeline) enforces the geometry itself.

`image_compare` is the other amplification path: it needs several float32
working arrays per pixel. It is computed one horizontal strip at a time, so
peak memory is O(rows_per_strip x width) rather than O(pixels) - measured at
~200 MB for a 16 MP pair, where the naive formulation needed ~2 GB and
extrapolated past 8 GB at the 64 MP input ceiling. A flat 64 MP PNG is only
~77 KiB, so the file-size cap alone would not have stopped it.
| `PICTOR_MAX_OUTPUT_PIXELS` | 256 MP | A requested resize to an absurd size; checked before allocating and after every pipeline step. |
| `PICTOR_MAX_CONCURRENCY` | 4 | Concurrent allocations. Size this with the container `mem_limit`. |
| `PICTOR_OP_TIMEOUT_SECONDS` | 120 | Runaway operations. |
| `PICTOR_MAX_BATCH_FILES` | 64 | Batch amplification. |
| `PICTOR_MAX_REQUEST_BODY_BYTES` | 48 MiB | HTTP request bodies (the SDK default of 4 MiB is too small for inline images). |

Truncated files are rejected rather than partially decoded
(`LOAD_TRUNCATED_IMAGES = False`), because a partially-populated image is silent
corruption.

### 3. Codec allow-list

Pillow can open roughly forty formats. This server accepts an explicit subset.

**Refused for input and output:** PDF, PostScript/EPS, WMF, HDF5, GRIB, FITS,
and the rest of the scientific/vector containers. They either shell out to an
external interpreter, are not images, or add a parser with an extensive CVE
history and no upside here.

**Refused for output:** PDF and EPS specifically — an image converter that will
write a PDF is a document-forgery primitive.

The format Pillow reports after opening the stream is authoritative. The file
extension, the caller's hint and the HTTP `Content-Type` are never used to
choose a decoder, so a file named `photo.png` whose bytes are a PostScript
program is rejected at the point Pillow identifies it.

### 4. SSRF (URL inputs)

Disabled unless `PICTOR_ALLOW_NET_FETCH=true`. When enabled, in order:

1. **Shape** — only `http`/`https`, no embedded credentials, no control
   characters, port must be in `PICTOR_FETCH_ALLOWED_PORTS` (default 80/443),
   host must match `PICTOR_FETCH_ALLOWED_HOSTS` when set.
2. **Address classification** — *every* address the name resolves to must be
   globally routable. Private, loopback, link-local, CGNAT, reserved, multicast
   and unique-local ranges are refused, as are IPv4-mapped forms of them.
   A round-robin answer mixing one public and one private address is refused
   wholesale, which is the trick that defeats most per-address filters.
   The ranges are listed explicitly rather than relying only on `ipaddress`
   predicates, because those change between Python releases — on CPython 3.14
   `IPv4Address("100.64.0.1").is_private` is `False`.
3. **IP pinning** — the connection is made to the validated literal address and
   never to the name, so a second resolution cannot happen. DNS rebinding is
   structurally impossible rather than merely unlikely. TLS still uses the
   hostname for SNI and certificate verification, so pinning costs nothing.
4. **Streaming cap** — the body is bounded while it is read, so a hostile server
   cannot exhaust memory by lying about `Content-Length` or streaming forever.
11. **Manual redirects** — each hop is re-validated from step 1, with a hard
   redirect ceiling. Automatic redirect following is how SSRF filters are
   bypassed in practice.
6. **No compression** — `Accept-Encoding: identity` keeps a decompressor out of
   the trust path and stops a small compressed body expanding without bound.
7. **No proxy inheritance** — a direct socket, so `HTTP_PROXY` cannot silently
   re-route the request.

`PICTOR_FETCH_VERIFY_TLS=false` disables certificate verification. It exists for
a pinned internal mirror and is a real downgrade; do not use it on the internet.

### 5. HTTP transport

| Control | Detail |
|---|---|
| Bind address | `127.0.0.1` by default. The compose port mapping is loopback-only. |
| Authentication | Optional pre-shared bearer token (also accepted as `X-API-Key`), compared with `secrets.compare_digest`. The comparison always runs, even with no token presented. |
| DNS rebinding | `Host` and `Origin` validated on **every** route by a single guard that wraps the whole app, including the MCP endpoint and the custom routes (health, file serving). The SDK's built-in check is deliberately switched off rather than fed the same list: its matcher understands only exact values and `name:*`, so it would silently reject patterns this server documents (`*`, `*.example.com`, a port-less `Host`) with a bare `421` — two policies, one of them weaker and unexplained. |
| Browser cross-site requests | A *present* `Origin` not on the allow-list is refused. Absent `Origin` is allowed, since non-browser MCP clients send none. |
| Missing `Host` | Refused. |
| Response headers | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cross-Origin-Resource-Policy: same-origin`, and `Cache-Control: no-store` on auth rejections. HSTS is opt-in (`PICTOR_ENABLE_HSTS`) and only meaningful behind TLS termination. |
| Proxy headers | `proxy_headers=False` and no `Server` header: trusting `X-Forwarded-For` from an untrusted client would let it spoof its origin in the logs. |
| Health endpoint | Information-free (`{"status":"ok"}`), the only auth-exempt route. |
| Signed output URLs | HMAC-SHA256 over the path and expiry, constant-time verification, expiry enforced. `/files/` is exempt from bearer auth *because* the signature is the credential — otherwise a browser `<img>` tag could never load a result. Serving is refused at startup unless a token or signing secret exists. |
| Served content type | Allow-listed to known image types. Anything else is `application/octet-stream` with `Content-Disposition: attachment`, and every response carries `Content-Security-Policy: default-src 'none'; sandbox`. A caller-supplied `output_name` has its extension forced to the encoder's own, so a JPEG cannot be stored as `.html`. Without this, attacker-controlled bytes (an ICC profile, an EXIF comment) could be rendered as HTML on the server's origin. |
| Access logging | uvicorn's per-request access log is **off** by default (`PICTOR_HTTP_ACCESS_LOG`), because a `/files/` URL carries its HMAC signature in the query string and a logged line is therefore a usable credential. Application logging is unaffected. |
| Error bodies | JSON-RPC shaped, and never include a stack trace or a host path. |

### 6. Container hardening

The compose file sets `read_only: true`, a 256 MB `/tmp` tmpfs, `cap_drop: ALL`,
`no-new-privileges`, `pids_limit: 256`, `mem_limit: 4g` and
`user: "${PUID:-1000}:${PGID:-1000}"`.

The runtime uid is a deployment setting rather than a fixed one, so that files
written to the mounted output directory belong to the operator instead of to an
unrelated account. This does not weaken the control: the process is still
unprivileged either way, and adopting a normal user's identity is in some
respects stricter than a dedicated uid, since that account already exists and
owns nothing the container can reach. `PUID=0`/`PGID=0` would defeat
`cap_drop`, `no-new-privileges` and the read-only root, so neither is ever
defaulted to root and both are documented as off-limits.

The image's own default identity is an unprivileged user with no shell and no
home directory, and `/data/output` is mode `1777` so that any uid may write it —
a mode restricted to the image's uid would make the configurable identity fail
at startup. The only writable paths are the output volume and the tmpfs.

The image ships **no credentials**, so publishing it leaks nothing.

### 7. Metadata and privacy

EXIF, GPS, ICC, XMP and embedded thumbnails are stripped from outputs by default
(`PICTOR_STRIP_METADATA`). `keep_icc=true` retains colour management without
retaining location data.

Transparency, frame durations and loop counts are deliberately *not* treated as
metadata: they change the pixels, and stripping them would corrupt output.

### 8. Supply chain

- `pyproject.toml` pins major versions (`mcp>=2.2,<3`, `pillow>=10.4`).
- The GPU stage installs torch with `--index-url`, not `--extra-index-url`.
  PyTorch documents the latter, but it puts two indexes in play and pip then
  takes the highest version across both, which is the dependency-confusion
  opening. PyTorch's index mirrors torch's own dependencies (sympy, networkx,
  filelock, jinja2, fsspec), so replacing PyPI entirely resolves cleanly and
  leaves only one index to trust. The CPU image — the default build target —
  never touches the CUDA index.
- `TORCH_INDEX_URL` selects the CUDA build and `TORCH_VERSION` optionally pins
  it; both are build arguments. The version is unpinned by default because a hard
  pin goes stale against a moving base image: `torch==2.4.1` stopped resolving
  the moment the base moved from Python 3.12 to 3.14, since the index it named
  carries no cp314 wheels at all. The default index is also chosen for hardware
  coverage, not only for wheels: `cu126` predates Blackwell, so it would install
  cleanly on an RTX 50-series card and then fail every kernel launch.
- The build context is minimal (`.dockerignore`), so no secret, test fixture or
  host file can reach a layer.
- The MCP SDK's own transport-security default is **DNS-rebinding protection
  disabled** for backwards compatibility. This server enables it explicitly, in
  `pictor_mcp.security.auth`, and turns the SDK's parallel check off so that one
  matcher decides. The guard is registered on the outer application and so also
  covers the SDK's own routes.

---

## Known limitations

Being explicit about these is more useful than implying they do not exist.

1. **No TLS.** The server speaks plain HTTP. Traffic is readable by anything on
   the path. Terminate TLS in a reverse proxy for anything but a trusted segment,
   and set `PICTOR_ENABLE_HSTS=true` when you do.
2. **A static bearer token is all-or-nothing.** There are no scopes, no
   per-user identity, no rotation, and no audit trail of which caller did what.
   Anyone with the token has the full tool surface.
3. **The token can be replayed.** No rate limiting or brute-force lockout on the
   auth endpoint. Use a long random token (32 bytes) and keep it off the network.
4. **The operation timeout is cooperative, not a hard kill.** A single
   CPU-bound Pillow call cannot be interrupted from Python without leaking the
   GIL or paying for a subprocess per operation. `PICTOR_OP_TIMEOUT_SECONDS`
   stops long *sequences* (pipeline steps, quality-search probes); one very slow
   decode or encode will still run to completion.
5. **Confinement depends on the mount.** The jail confines paths *inside the
   container*. If you mount a sensitive host directory into `/data/input`, the
   server will read it. Mount narrowly and read-only.
6. **`image_batch` can process many files.** Bounded by `PICTOR_MAX_BATCH_FILES`,
   but 64 large images still consume real CPU and memory.
7. **ML background removal loads a model.** The `ml` image adds ~1 GB and an
   ONNX runtime. A model file is code-adjacent input; treat image updates like
   any other dependency change.
8. **Signed URLs are bearer credentials.** Anyone holding one can fetch that file
   until it expires. TTL is configurable; treat a logged URL as a leaked
   credential.
9. **No malware scanning.** The server will happily convert an image that an
   antivirus would flag. It processes pixels, not intent.
10. **Metadata stripping is best-effort.** It removes what Pillow exposes. It is
   not a forensic anonymiser, and unusual private tags in exotic formats may
   survive.
11. **The `--check` flag prints resolved configuration.** Secrets are redacted,
    but paths and limits are shown. Treat its output as internal.

---
## Deployment checklist

- [ ] `PICTOR_AUTH_TOKEN` set to 32 random bytes (`openssl rand -hex 32`).
- [ ] Port published to loopback, or to a specific interface — never `0.0.0.0`
      on an untrusted network.
- [ ] TLS terminated in front, if the network is not trusted.
- [ ] `PICTOR_ALLOWED_HOSTS` / `PICTOR_ALLOWED_ORIGINS` set to the real external
      address if you changed the bind or published port. The default also accepts
      the compose service name (`pictor-mcp:*`) so sibling containers work; that
      name is not resolvable from a browser, so it adds no reachability.
- [ ] `PICTOR_INPUT_ROOTS` mounted read-only and as narrowly as possible.
- [ ] `PICTOR_OUTPUT_ROOT` on a host directory you are happy to grow.
- [ ] `PICTOR_ALLOW_NET_FETCH` left `false` unless URL inputs are genuinely
      needed; if enabled, set `PICTOR_FETCH_ALLOWED_HOSTS`.
- [ ] `PICTOR_MAX_CONCURRENCY` × `PICTOR_MAX_PIXELS` consistent with `mem_limit`.
- [ ] Container hardening intact (`read_only`, `cap_drop`, `no-new-privileges`,
      non-root user).
- [ ] `PUID`/`PGID` set to your own `id -u`/`id -g`, or the host directories
      `chown`ed to match — and neither left at `0`.
- [ ] Backups or retention: the output volume grows without bound.

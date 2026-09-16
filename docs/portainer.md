# Building pictor-mcp locally, and deploying it with Portainer

The published CUDA image is several gigabytes, and most of that is PyTorch's
NVIDIA wheels. Pulling it again for every change is the expensive part of working
on this repository, and it is avoidable: build the image **on the host that runs
it**, and let Portainer deploy a stack that names that local image. Nothing is
pushed to a registry, and nothing is pulled.

This document covers what Portainer will and will not do, and the three ways to
wire it up. It assumes Portainer Business Edition, which is where stack webhooks
and the `pullimage=false` parameter live.

---

## 1. What Portainer does, and does not, do

Worth being precise, because "Portainer workflows can build on a GitLab push" is
a reasonable reading of the feature list and it is not quite what happens.

| Portainer feature | What it actually does |
|---|---|
| **Workflows / App Delivery** | A GitOps view over stacks deployed **from a Git repository**. In the versions documented at the time of writing (2.39 LTS, 2.44/2.45) a workflow pulls a compose file or manifest from a source and deploys it to **edge groups**. It deploys an image; it does not build one. |
| **Stacks from Git** | Portainer's own FAQ is explicit: building images from `docker-compose.yml` in a Git repository is *not fully implemented*. "If the image is built separately and referenced from docker-compose, it should install without an issue." |
| **GitOps updates** | Portainer can poll a repository, or expose a **webhook** you call from CI to make it re-fetch and redeploy. That is the GitLab hook: it triggers a *deploy*, not a build. |
| **Stack webhook** (BE) | `POST /api/stacks/webhooks/<id>` redeploys a stack. With `?pullimage=false` it uses the image already on the host instead of pulling - which is what makes a locally built image deployable. |
| **Images → Build a new image** | Portainer *can* build: from a Dockerfile in its web editor, an uploaded Dockerfile, or a URL to a tarball or public GitHub repository. It has no git-push trigger and its UI does not expose the Dockerfile stage, but it calls the Docker build API, which Portainer proxies verbatim - so `target`, `buildargs` and `nocache` all work when you call that API yourself. |

So the shape of the solution is: **the build happens where the Docker daemon
is** - your laptop, a shell on the host, a GitLab runner with the host socket, or
Portainer's own API - and **Portainer deploys the result**.

Two consequences worth internalising:

* The image must exist on the Docker host that will run it. On a Swarm or
  multi-node environment, build on each node or push to a registry the nodes can
  reach (route D below).
* Nothing in the deployment path contacts a registry: the image is built on the
  host, and the stack names it by a tag no registry has. If a pull is attempted
  it fails loudly rather than quietly downloading gigabytes.

---

## 2. Route A - build on the host, deploy with a stack (recommended)

This is the no-registry, no-pull path, and the one to start with.

**On the Docker host** (SSH, or a shell on the machine Portainer manages):

```bash
git clone https://github.com/TheRealChickenlegs/pictor-mcp.git
cd pictor-mcp

make build-gpu          # or: scripts/build_image.sh --target gpu
```

`make build-gpu` builds the `gpu` stage, tags it `pictor-mcp:local-gpu`, and then
starts that image with `--network none` to prove it runs offline. The first build
downloads the CUDA wheels; after that a source edit rebuilds one small layer,
because `COPY src/` sits above every dependency layer in the Dockerfile.

**In Portainer**, create the stack from this repository:

1. **Stacks → Add stack → Git repository.** Pick your GitLab/GitHub source, the
   branch, and `docker-compose.yml` as the compose path.
2. **Additional paths → Add file:** `docker-compose.gpu.yml` (or
   `docker-compose.ml.yml`). This is the equivalent of a second `-f`, which is
   how the GPU and ML variants are selected.
3. **Environment variables** - the ones that matter here:

   ```
   IMAGE_REPO=pictor-mcp
   IMAGE_TAG_GPU=local-gpu
   PUID=1000
   PGID=1000
   PICTOR_AUTH_TOKEN=<openssl rand -hex 32>
   ```

   `IMAGE_REPO` + `IMAGE_TAG_GPU` are what point the stack at the locally built
   image instead of `ghcr.io/...:gpu`. Use `IMAGE_TAG=local` for the CPU image and
   `IMAGE_TAG_ML=local-ml` for the ML one. Nothing is pulled, because
   `pictor-mcp:local-gpu` does not exist in any registry - if a pull is attempted,
   it fails loudly rather than silently downloading something.
4. **Deploy the stack.**

**Iterating.** After a change:

```bash
make build-gpu
PORTAINER_WEBHOOK_URL=https://portainer.internal:9443/api/stacks/webhooks/<id> \
  make deploy
```

`scripts/portainer_deploy.sh` appends `pullimage=false`, so the redeploy picks up
the image you just built. The webhook answers before the deployment finishes; the
stack's log in Portainer is where the outcome is.

---

## 3. Route B - let Portainer run the build

If you would rather not have a checkout on the host, Portainer's daemon can do
the build itself. It needs an API token (**My account → Access tokens**) and a
checkout of the repository wherever you run the script from:

```bash
export PORTAINER_URL=https://portainer.internal:9443
export PORTAINER_API_TOKEN=ptr_...
make portainer-build BUILD_TARGET=gpu
```

That streams the build context - only `Dockerfile`, `pyproject.toml`,
`README.md`, `src/` and `.dockerignore`, never `.env`, `output/` or the
`.venv` - to `POST /api/endpoints/<id>/docker/build`, renders the build output as
it arrives, and then confirms the tag appears in Portainer's image list.

`PORTAINER_INSECURE=1` handles a self-signed certificate, `PORTAINER_ENDPOINT_ID`
picks the environment (default `1`), and `TORCH_INDEX_URL` / `TORCH_VERSION` are
forwarded as Docker build arguments.

The equivalent in the UI is **Images → Build a new image → Upload → URL**, with a
tarball URL and the Dockerfile path inside it. It works for the CPU image; for
`gpu`/`ml` you need the API, because the UI has no field for the build stage.

---

## 4. Route C - GitLab CI builds it on push

`.gitlab-ci.yml` in this repository runs the whole loop: checks, then a build into
the runner's Docker daemon, then (once you configure it) the Portainer webhook.

Requirements, in order of preference:

* **A shell runner on the Docker host**, or **a docker executor with
  `/var/run/docker.sock` mounted**, or `DOCKER_HOST` pointing at a remote/TLS
  daemon. Then `build:gpu` produces an image the deployment can use directly.
* **Docker-in-docker** also works, but the daemon is thrown away with the job, so
  set `PUSH_IMAGE=1` and a registry (see route D).

Set these in **Settings → CI/CD → Variables**:

| Variable | Why |
|---|---|
| `PORTAINER_WEBHOOK_URL` | The stack webhook. Without it the `deploy:stack` job is skipped. |
| `PORTAINER_INSECURE` | `1` for a self-signed Portainer certificate. |
| `DOCKER_HOST` | If the daemon is not on the runner itself, e.g. `tcp://10.0.0.5:2376`. |
| `PUSH_IMAGE` | `1` to enable `build:push` to the GitLab container registry. |

`build:cpu` runs on every branch. `build:gpu` and `build:ml` run automatically on
the default branch and on tags, and are a manual click elsewhere - expensive jobs
should not follow every work-in-progress commit. Deploy is **manual** by default;
delete `when: manual` in the job to make a push to the default branch deploy
itself.

If your GitLab is a mirror of this GitHub repository rather than the source of
truth, the pipeline files still apply - push the mirror and GitLab CI does the
rest.

---

## 5. Why rebuilds are cheap now

Two changes in the `Dockerfile`, and they are the reason this is worth setting
up:

1. **The application is installed last.** `COPY src/` used to sit *below* the
   PyTorch install, so every source edit invalidated the CUDA layer and the next
   build re-fetched the NVIDIA wheels - locally, in CI, and through Portainer's
   build API. The published stages (`base`, `gpu`, `ml`) now inherit from
   `base-deps`, `gpu-deps` and `ml-deps`, and only they copy the source.
   `TestImageLayerOrder` in `tests/test_deployment_config.py` fails if anyone
   moves a `COPY` back down.
2. **Every install has a pip cache mount.** On a persistent daemon the wheels
   stay on disk, so even a rebuild that *does* touch the dependency layer - a
   `pyproject.toml` change, say - installs from the local cache instead of
   downloading gigabytes again. The cache is a BuildKit mount, never an image
   layer, so it does not ship.

This is also why a GitHub-published image no longer changes 6 GB of layers when
you edit a Python file: the CUDA layer is now shared across releases.

---

## 6. Route D - a registry, when the deployment host is not the build host

If Portainer manages a different machine from the one that builds, the image has
to travel. Prefer a registry on the LAN over GHCR:

```bash
docker run -d -p 5000:5000 --restart unless-stopped --name registry registry:2

scripts/build_image.sh --target ml --tag registry.lan:5000/pictor-mcp:local-ml --push
```

Then set `IMAGE_REPO=registry.lan:5000/pictor-mcp` and `IMAGE_TAG_ML=local-ml` on
the stack. Docker will re-pull, but only the layers that changed, and the multi-
gigabyte dependency layers are already on the host and are reused.

Add the registry in Portainer (**Registries → Add registry → Custom**) if it needs
credentials.

---

## 7. Troubleshooting

| Symptom | Cause |
|---|---|
| Stack logs `manifest unknown` / `pull access denied` for `pictor-mcp:local-gpu` | The image is not on the host that is deploying. Build on that host (route A/B), or switch to route D. |
| Portainer rebuilds nothing and the old image keeps running | The webhook pulled and found the old tag, or the stack redeployed without `pullimage=false`. Use `scripts/portainer_deploy.sh`, which sets it. |
| `docker build` fails with `the --mount option requires BuildKit` | The daemon is older than Docker 23, or BuildKit is switched off. `--mount=type=cache` is the pip cache; set `DOCKER_BUILDKIT=1`, or upgrade Docker. |
| Build succeeds but the container restarts with an import error | The application layer did not replace the dependency stage's placeholder package. The Dockerfile has a build-time import check for exactly this, so it should fail the build instead - if it did not, the build output is the thing to read. |
| `deploy:stack` never runs | `PORTAINER_WEBHOOK_URL` is not set, or the branch is not the default branch. |
| Portainer's workflow does not show the stack | Workflows lists stacks deployed **from a Git source**. A stack created in the web editor or by upload is not a workflow. |

## 8. What is not verified here

* The Dockerfile's new stage layout and its cache mounts have not been built in
  the environment this repository was developed in (no Docker daemon there). The
  static contract tests in `tests/test_deployment_config.py` pin the stage graph,
  the layer order and the mount/cache-directory pairing, and
  `scripts/build_image.sh` verifies each build by starting it - but the first
  real build is still the first real build. Expect to run it once and read the
  output.
* The Portainer workflows described above were read from Portainer's
  documentation and, for the build proxy, its source. Your Portainer version may
  differ; the API path (`/api/endpoints/<id>/docker/build`) and the stack webhook
  have been stable for a long time, which is why the scripts use them rather than
  the UI.

"""Contract tests between the deployment files and the config parser.

`docker-compose*.yml` and `.env.example` are long lists of variable names that
must match `config.py` exactly. Drift here is silent and dangerous: a renamed
variable leaves the compose value ignored and the *code* default in force, which
is not always the secure one.

These tests do not need Docker. They emulate Compose's ``${VAR:-default}``
interpolation with the variable unset - which is exactly what a bare
``docker compose up`` does - and feed the result through the real parser. They
also emulate overlay merging, so a layer that silently breaks a setting is
caught here rather than at deploy time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from pictor_mcp.config import Config, inert_allow_list_entries, load_config

try:  # Python 3.11+
    # `# novermin` is required: the version-floor check cannot see through the
    # try/except and would report this module as needing 3.11. The fallback is
    # what makes it 3.10-compatible, and the checker cannot know that.
    import tomllib  # novermin
except ModuleNotFoundError:  # pragma: no cover - exercised by the 3.10 CI job
    # tomllib only landed in 3.11 and 3.10 is the declared floor, so the version
    # matrix runs this module there. tomli is the same parser under its former
    # name, declared as a dev dependency for exactly that case.
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent
BASE_COMPOSE = ROOT / "docker-compose.yml"
GPU_OVERLAY = ROOT / "docker-compose.gpu.yml"
ML_OVERLAY = ROOT / "docker-compose.ml.yml"
BUILD_OVERLAY = ROOT / "docker-compose.build.yml"
ALL_COMPOSE = (BASE_COMPOSE, GPU_OVERLAY, ML_OVERLAY, BUILD_OVERLAY)
ENV_EXAMPLE = ROOT / ".env.example"
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"

SERVICE = "pictor-mcp"

#: Fallback registry path, used only when the git remote cannot be read (a
#: source tarball, for instance). The authority is the remote itself; see
#: :func:`expected_image_repo`.
FALLBACK_IMAGE_REPO = "ghcr.io/therealchickenlegs/pictor-mcp"

_INTERPOLATION = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")

#: Variables consumed by Compose itself rather than by the server. They
#: deliberately have no PICTOR_ prefix so they cannot be confused with (or
#: accidentally documented as) server settings.
COMPOSE_ONLY_VARS = frozenset(
    {
        "IMAGE_REPO",
        "IMAGE_TAG",
        "IMAGE_TAG_GPU",
        "IMAGE_TAG_ML",
        "BUILD_TARGET",
        "LOCAL_IMAGE_TAG",
        "BIND_ADDRESS",
        "PUID",
        "PGID",
    }
)

#: Variables consumed by a bundled third-party library rather than by this
#: project: rembg resolves its model directory from U2NET_HOME.
THIRD_PARTY_VARS = frozenset({"U2NET_HOME"})

#: Keys that must come from the shared hardening anchor rather than being
#: redefined per file, so an overlay cannot silently drop one.
HARDENING_KEYS = frozenset({"read_only", "cap_drop", "security_opt", "pids_limit", "mem_limit", "user"})


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _merge(*paths: Path) -> dict:
    """Merge ``services.<name>`` the way `docker compose -f` does.

    Compose merges rather than replaces: ``environment`` maps merge per key, and
    list-valued keys such as ``volumes`` accumulate. A shallow dict update would
    discard the base environment the moment an overlay adds one variable, which
    is precisely the mistake this helper exists to catch.
    """
    merged: dict = {"services": {}}
    for path in paths:
        for name, service in (_load(path).get("services") or {}).items():
            target = merged["services"].setdefault(name, {})
            for key, value in (service or {}).items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    target[key] = {**target[key], **value}
                elif key in {"volumes", "ports"} and isinstance(value, list):
                    target[key] = [*(target.get(key) or []), *value]
                else:
                    target[key] = value
    return merged


def _interpolate(value: object) -> object:
    """Apply Compose's default-value interpolation with every variable unset."""
    if not isinstance(value, str):
        return value
    return _INTERPOLATION.sub(lambda match: match.group(2) or "", value)


def _env(*paths: Path) -> dict[str, str]:
    services = _merge(*paths)["services"]
    assert SERVICE in services, f"{SERVICE} not defined by {[p.name for p in paths]}"
    raw = services[SERVICE].get("environment") or {}
    assert isinstance(raw, dict), f"{SERVICE}.environment did not merge into a mapping"
    return {key: _interpolate(value) for key, value in raw.items()}  # type: ignore[misc]


def _config(**overrides: str) -> Config:
    """The compose defaults, with individual settings replaced.

    Composing an override on top of the shipped file rather than loading a bare
    mapping keeps these tests honest about the deployment people actually get:
    a setting that only looks fine in isolation fails here.
    """
    return load_config({**_env(BASE_COMPOSE), **overrides})


def _env_example_assignments() -> dict[str, str]:
    """Active (uncommented) ``KEY=value`` lines from the template."""
    values: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


def expected_image_repo() -> str:
    """Derive the GHCR path from the git remote.

    Deliberately not a constant. A hard-coded expectation is a second copy of the
    same string, so a typo in the compose files can be mirrored in the assertion
    and the test passes while every deployment pulls an image that does not
    exist - which is exactly what happened here: the owner was spelled
    "therealcickenlegs" in five files and in the expectation, so the check
    agreed with the bug. The remote is the one thing that cannot be wrong about
    where the images live.
    """
    import subprocess

    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return FALLBACK_IMAGE_REPO

    match = re.search(r"github\.com[:/]([^/]+)/([^/\s]+?)(?:\.git)?$", url)
    if not match:  # pragma: no cover - remote is not GitHub
        return FALLBACK_IMAGE_REPO
    # GHCR lower-cases the owner and repository when it namespaces a package.
    return f"ghcr.io/{match.group(1).lower()}/{match.group(2).lower()}"


def _pyproject() -> dict[str, Any]:
    """Parse pyproject.toml, working on 3.10 as well as 3.11+."""
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def _config_names() -> set[str]:
    """Every PICTOR_ name the code actually reads."""
    source = (ROOT / "src" / "pictor_mcp" / "config.py").read_text()
    return set(re.findall(r'"(PICTOR_[A-Z0-9_]+)"', source))


def _interpolated_names(path: Path) -> set[str]:
    """Every ``${VAR}`` used in a compose file, ignoring commented-out lines.

    Comments discuss alternatives such as ``${VAR:-default}``, and scanning them
    would report variables that are documented rather than used.
    """
    active = "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("#"))
    return {name for name, _ in _INTERPOLATION.findall(active)}


_STAGE_HEADER = re.compile(r"^FROM\s+(\S+)\s+AS\s+(\S+)\s*$", re.MULTILINE)


def _docker_stages() -> dict[str, tuple[str, str]]:
    """Map each stage alias to ``(base, body)``, in file order.

    The build stages are not independent: `gpu` inherits `gpu-deps`, which
    inherits `base-deps`, and the layer order across that chain is what decides
    whether a source edit re-downloads the CUDA wheels. Answering that question
    needs the body of a specific stage, which is what this returns.
    """
    text = DOCKERFILE.read_text()
    headers = list(_STAGE_HEADER.finditer(text))
    stages: dict[str, tuple[str, str]] = {}
    for index, match in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        stages[match.group(2)] = (match.group(1), text[match.end() : end])
    return stages


def _stage_commands(alias: str) -> str:
    """A stage's body with its comments removed.

    Comments here explain the approach that was *rejected* - "do not use
    --extra-index-url", "the placeholder is why the app is installed last" - so a
    naive `in` check would match the explanation instead of the code.
    """
    _, body = _docker_stages()[alias]
    return "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))


def _run_blocks(body: str) -> list[str]:
    """One string per RUN instruction, continuations joined.

    Per-instruction granularity is what makes "every pip install has a cache
    mount" answerable: a stage-wide search would be satisfied by a mount attached
    to a different command.
    """
    joined = re.sub(r"\\\s*\n", " ", body)
    return [line.strip() for line in joined.splitlines() if line.lstrip().startswith("RUN ")]


class TestComposeStructure:
    @pytest.mark.parametrize("path", ALL_COMPOSE, ids=lambda p: p.name)
    def test_file_parses_as_yaml_with_services(self, path: Path) -> None:
        document = _load(path)
        assert isinstance(document, dict)
        assert document.get("services"), f"{path.name} defines no services"

    @pytest.mark.parametrize("path", ALL_COMPOSE, ids=lambda p: p.name)
    def test_every_file_targets_the_same_service(self, path: Path) -> None:
        """Overlays only make sense if they extend the same service name."""
        assert set(_load(path)["services"]) == {SERVICE}, path.name

    def test_base_compose_has_exactly_one_service(self) -> None:
        # One service is what makes `docker compose up` unambiguous and keeps the
        # variants from fighting over the same published port.
        assert list(_load(BASE_COMPOSE)["services"]) == [SERVICE]

    @pytest.mark.parametrize("path", [BASE_COMPOSE, GPU_OVERLAY, ML_OVERLAY], ids=lambda p: p.name)
    def test_runs_a_published_image_and_does_not_build(self, path: Path) -> None:
        """The deployment path must pull an image, not compile one.

        `build` is deliberately absent so `docker compose up` is a pull; the
        contributor path is the separate build overlay.
        """
        service = _load(path)["services"][SERVICE]
        assert "build" not in service, f"{path.name} should not build"
        assert service["image"].startswith("${IMAGE_REPO"), service["image"]

    def test_only_the_build_overlay_builds(self) -> None:
        assert "build" in _load(BUILD_OVERLAY)["services"][SERVICE]
        for path in (BASE_COMPOSE, GPU_OVERLAY, ML_OVERLAY):
            assert "build" not in _load(path)["services"][SERVICE], path.name

    @pytest.mark.parametrize("path", [BASE_COMPOSE, GPU_OVERLAY, ML_OVERLAY], ids=lambda p: p.name)
    def test_image_path_matches_the_git_remote(self, path: Path) -> None:
        """The registry path must be the one the remote actually implies.

        Compared against the git remote rather than a constant, so a typo in the
        owner or repository name cannot be mirrored in the expectation. Getting
        this wrong means every deployment pulls a non-existent image, and the
        failure surfaces on the operator's machine rather than in CI.
        """
        expected = expected_image_repo()
        image = _load(path)["services"][SERVICE]["image"]
        assert f"${{IMAGE_REPO:-{expected}}}" in image, f"{path.name}: {image}"

    def test_the_registry_path_is_consistent_everywhere(self) -> None:
        """Every file that repeats the path must agree with the remote."""
        expected = expected_image_repo()
        seen: set[str] = set()
        for path in (*ALL_COMPOSE, ENV_EXAMPLE, ROOT / "README.md"):
            seen.update(re.findall(r"ghcr\.io/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", path.read_text()))
        assert seen == {expected}, f"inconsistent registry paths: {sorted(seen)}"

    @pytest.mark.parametrize("path", sorted((ROOT / ".github" / "workflows").glob("*.yml")), ids=lambda p: p.name)
    def test_image_references_built_in_shell_are_lowercased(self, path: Path) -> None:
        """GHCR rejects any uppercase character in a repository path.

        `github.repository` keeps the owner's capitalisation
        (`TheRealChickenlegs/pictor-mcp`), and `docker/metadata-action` lowercases
        the images it generates, so the push succeeds. A reference assembled by
        hand in a `run:` step does not, and fails with "repository name must be
        lowercase" - which is exactly how the verify job broke.
        """
        document = yaml.safe_load(path.read_text())
        for job in (document.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                script = step.get("run")
                if not script or "GITHUB_REPOSITORY" not in script:
                    continue
                active = "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))
                lowercased = "tr '[:upper:]' '[:lower:]'" in active or "${GITHUB_REPOSITORY,,}" in active
                assert lowercased, f"{path.name}: {step.get('name')!r} uses GITHUB_REPOSITORY without lowercasing it"
                # And the reference must be built from the lowercased copy.
                assert "${REGISTRY}/${GITHUB_REPOSITORY}" not in active, (
                    f"{path.name}: {step.get('name')!r} builds an image reference from the raw name"
                )

    def test_the_publish_workflow_pushes_to_the_same_registry(self) -> None:
        """The workflow derives the repository from the checkout, not a literal."""
        workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text()
        assert "REGISTRY: ghcr.io" in workflow
        assert "images: ${{ env.REGISTRY }}/${{ github.repository }}" in workflow

    @pytest.mark.parametrize(
        ("path", "tag_var", "tag_default"),
        [(GPU_OVERLAY, "IMAGE_TAG_GPU", "gpu"), (ML_OVERLAY, "IMAGE_TAG_ML", "ml")],
    )
    def test_image_overlays_switch_the_tag(self, path: Path, tag_var: str, tag_default: str) -> None:
        image = _load(path)["services"][SERVICE]["image"]
        assert f"${{{tag_var}:-{tag_default}}}" in image, image

    def test_base_uses_the_cpu_tag(self) -> None:
        image = _load(BASE_COMPOSE)["services"][SERVICE]["image"]
        assert "${IMAGE_TAG:-latest}" in image, image

    @pytest.mark.parametrize("path", [GPU_OVERLAY, ML_OVERLAY], ids=lambda p: p.name)
    def test_cuda_overlays_request_the_gpu(self, path: Path) -> None:
        assert _load(path)["services"][SERVICE].get("gpus") == "all"

    def test_base_does_not_request_the_gpu(self) -> None:
        assert "gpus" not in _load(BASE_COMPOSE)["services"][SERVICE]


class TestComposeConfiguration:
    @pytest.mark.parametrize(
        "extra",
        [(), (GPU_OVERLAY,), (ML_OVERLAY,), (BUILD_OVERLAY,)],
        ids=["cpu", "gpu", "ml", "build"],
    )
    def test_merged_defaults_are_valid_configuration(self, extra: tuple[Path, ...]) -> None:
        """A bare `docker compose up` must produce a config the server accepts."""
        config = load_config(_env(BASE_COMPOSE, *extra))
        assert config.transport == "streamable-http"

    def test_cpu_deployment_disables_the_gpu(self) -> None:
        """The CPU image has no torch, so asking for it only wastes startup time."""
        assert load_config(_env(BASE_COMPOSE)).gpu == "off"

    @pytest.mark.parametrize("overlay", [GPU_OVERLAY, ML_OVERLAY], ids=lambda p: p.name)
    def test_cuda_overlays_request_auto_detection(self, overlay: Path) -> None:
        assert load_config(_env(BASE_COMPOSE, overlay)).gpu == "auto"

    @pytest.mark.parametrize(
        "extra",
        [(), (GPU_OVERLAY,), (ML_OVERLAY,)],
        ids=["cpu", "gpu", "ml"],
    )
    def test_secure_defaults_survive_every_overlay(self, extra: tuple[Path, ...]) -> None:
        config = load_config(_env(BASE_COMPOSE, *extra))
        assert config.fetch.enabled is False, "URL fetching must be opt-in"
        assert config.http.serve_outputs is False, "output serving must be opt-in"
        assert config.strip_metadata is True
        assert config.http.dns_rebinding_protection is True
        assert config.http.access_log is False
        assert config.limits.max_pixels > 0

    @pytest.mark.parametrize("path", [BASE_COMPOSE, GPU_OVERLAY, ML_OVERLAY], ids=lambda p: p.name)
    def test_hardening_is_present_after_merging(self, path: Path) -> None:
        service = _merge(BASE_COMPOSE, path)["services"][SERVICE]
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in service["security_opt"]
        # Not a literal: the container identity is a deployment choice so that
        # files in ./output belong to the operator rather than a stranger.
        assert service["user"].startswith("${PUID"), service["user"]
        assert service["tmpfs"], "a read-only root needs a writable /tmp"
        assert service["pids_limit"] and service["mem_limit"]
        assert service["healthcheck"]["test"]

    def test_base_inherits_hardening_from_the_shared_anchor(self) -> None:
        """One anchor means a variant cannot quietly lose a hardening setting.

        PyYAML resolves the ``<<`` merge key while parsing, so "is this value
        inherited?" has to be answered from the source text rather than from the
        loaded document.
        """
        text = BASE_COMPOSE.read_text()
        assert "<<: *pictor-hardening" in text
        assert "volumes: *pictor-volumes" in text
        assert "environment: *pictor-environment" in text

    @pytest.mark.parametrize("path", [GPU_OVERLAY, ML_OVERLAY], ids=lambda p: p.name)
    def test_overlays_never_redeclare_a_hardening_key(self, path: Path) -> None:
        """An overlay may only add; redefining one key would drop the others."""
        service = _load(path)["services"][SERVICE]
        assert not (HARDENING_KEYS & set(service)), (
            f"{path.name} redefines hardening keys: {sorted(HARDENING_KEYS & set(service))}"
        )

    @pytest.mark.parametrize(
        ("puid", "pgid"),
        [("1000", "1000"), ("1001", "1001"), ("1000", "100")],
    )
    def test_the_container_runs_as_the_configured_uid_and_gid(
        self, puid: str, pgid: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Files in ./output must belong to the operator, not to an unrelated uid."""
        monkeypatch.setenv("PUID", puid)
        monkeypatch.setenv("PGID", pgid)
        # Compose resolves ${VAR} from the environment before the container sees
        # it, so this asserts the value an operator would actually get.
        resolved = re.sub(
            r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}",
            lambda m: os.environ.get(m.group(1), m.group(2) or ""),
            _load(BASE_COMPOSE)["services"][SERVICE]["user"],
        )
        assert resolved == f"{puid}:{pgid}"

    def test_the_default_identity_is_unprivileged(self) -> None:
        """Root would negate cap_drop, no-new-privileges and the read-only root."""
        default = re.search(r"\$\{PUID:-([^}]*)\}", _load(BASE_COMPOSE)["services"][SERVICE]["user"])
        assert default, "user is not configurable"
        assert default.group(1) not in {"", "0"}, default.group(1)
        gid = re.search(r"\$\{PGID:-([^}]*)\}", _load(BASE_COMPOSE)["services"][SERVICE]["user"])
        assert gid and gid.group(1) not in {"", "0"}

    def test_the_default_host_list_accepts_the_compose_service_name(self) -> None:
        """The sibling-container case must work without reading the docs.

        Another container on the same Docker network reaches this one as
        ``http://pictor-mcp:8077/mcp``, so the Host header is the service name.
        A loopback-only default rejects that with "host header is not allowed",
        which reads as an auth failure but is not one - the operator has no way
        to guess that a *name* is the missing piece. Renaming the service must
        keep this passing, so the expected value is derived from the compose
        file rather than hardcoded.
        """
        from pictor_mcp.security.auth import _host_matches

        config = load_config(_env(BASE_COMPOSE))
        assert f"{SERVICE}:*" in config.http.allowed_hosts, config.http.allowed_hosts
        assert _host_matches(f"{SERVICE}:{config.http.port}", config.http.allowed_hosts) is True
        # A port-less Host is legal HTTP and must reach the same place.
        assert _host_matches(SERVICE, config.http.allowed_hosts) is True
        # Widening the default must not have widened it to everything.
        assert _host_matches("evil.example.com", config.http.allowed_hosts) is False
        assert _host_matches("pictor-mcp.evil.com", config.http.allowed_hosts) is False

    @pytest.mark.parametrize("path", [BASE_COMPOSE, GPU_OVERLAY, ML_OVERLAY], ids=lambda p: p.name)
    def test_no_service_hardcodes_a_runtime_uid(self, path: Path) -> None:
        """A literal here is what made the container and the host disagree."""
        service = _load(path)["services"][SERVICE]
        for key in ("user",):
            if key in service:
                assert "${PUID" in str(service[key]), f"{path.name}: {service[key]}"

    def test_the_image_output_directory_is_writable_by_any_uid(self) -> None:
        """The image must not assume its own uid, since compose overrides it."""
        text = DOCKERFILE.read_text()
        assert re.search(r"chmod 1?777 /data/output", text), "output dir is uid-specific"

    def test_ports_are_loopback_only_by_default(self) -> None:
        """The default deployment must not be reachable off-host."""
        ports = _load(BASE_COMPOSE)["services"][SERVICE]["ports"]
        active = [str(entry) for entry in ports if not str(entry).strip().startswith("#")]
        assert active, "no active port mapping"
        for mapping in active:
            assert mapping.startswith("${BIND_ADDRESS:-127.0.0.1}:"), mapping

    def test_ports_document_how_to_bind_all_interfaces(self) -> None:
        """The LAN path is the one change users actually need to find."""
        text = BASE_COMPOSE.read_text()
        assert "ALL INTERFACES" in text
        assert re.search(r"#\s+- \"\$\{PICTOR_PORT:-8077\}", text), "all-interfaces mapping missing"
        assert "0.0.0.0" in text, "the BIND_ADDRESS=0.0.0.0 alternative is not mentioned"
        assert "192.168.1.10" in text, "the single-interface alternative is not mentioned"

    def test_bind_address_is_documented_in_the_template(self) -> None:
        assert "BIND_ADDRESS=127.0.0.1" in ENV_EXAMPLE.read_text()

    @pytest.mark.parametrize("path", ALL_COMPOSE, ids=lambda p: p.name)
    def test_input_is_mounted_read_only(self, path: Path) -> None:
        service = _merge(BASE_COMPOSE, path)["services"][SERVICE]
        mounts = service.get("volumes") or []
        resolved = {
            str(entry).split(":")[1]: str(entry)
            for entry in mounts
            if not str(entry).strip().startswith("#") and ":" in str(entry)
        }
        assert resolved.get("/data/input", "").endswith(":ro"), resolved

    def test_environment_is_shared_by_the_overlays(self) -> None:
        """An anchor keeps a security setting from applying to only one variant."""
        base = _env(BASE_COMPOSE)
        assert base, "the base compose defines no environment"
        for overlay in (GPU_OVERLAY, ML_OVERLAY):
            overlaid = _env(BASE_COMPOSE, overlay)
            differing = {key for key in base if base[key] != overlaid.get(key)}
            # These are the only keys an overlay is allowed to change.
            differing -= {"PICTOR_GPU", "PICTOR_BG_MODEL", "U2NET_HOME"}
            assert not differing, f"{overlay.name} diverges on {sorted(differing)}"

    def test_compose_does_not_duplicate_the_environment_block(self) -> None:
        """The environment lives in one anchor; copies are how settings drift."""
        text = BASE_COMPOSE.read_text()
        assert text.count("x-pictor-environment:") == 1
        assert text.count("*pictor-environment") == 1
        assert text.count("x-pictor-hardening:") == 1
        assert text.count("*pictor-hardening") == 1


class TestVariableCoverage:
    def test_env_example_documents_every_supported_variable(self) -> None:
        """Every setting must appear, even if only as a commented-out template.

        Secrets and override-only settings are deliberately shipped commented so
        that copying the template verbatim cannot enable something surprising.
        """
        text = ENV_EXAMPLE.read_text()
        missing = {name for name in _config_names() if name not in text}
        assert not missing, f"undocumented settings: {sorted(missing)}"

    def test_env_example_invents_no_server_settings(self) -> None:
        unknown = {key for key in _env_example_assignments() if key.startswith("PICTOR_")} - _config_names()
        assert not unknown, f"names the code does not read: {sorted(unknown)}"

    def test_env_example_documents_the_compose_variables(self) -> None:
        """The Compose-only knobs are part of the documented surface too."""
        text = ENV_EXAMPLE.read_text()
        missing = {name for name in COMPOSE_ONLY_VARS if f"{name}=" not in text}
        assert not missing, f"undocumented Compose settings: {sorted(missing)}"

    def test_env_example_defaults_are_valid(self) -> None:
        """Copying the template verbatim must give a working configuration."""
        load_config(_env_example_assignments())

    def test_compose_forwards_every_supported_variable(self) -> None:
        missing = _config_names() - set(_env(BASE_COMPOSE))
        assert not missing, f"not forwarded by compose: {sorted(missing)}"

    @pytest.mark.parametrize("path", ALL_COMPOSE, ids=lambda p: p.name)
    def test_every_interpolated_variable_is_declared(self, path: Path) -> None:
        """Catch a typo'd ``${VAR}`` that would silently fall back to a default.

        Compose substitutes an empty string for an undefined variable, so a
        misspelled name does not error - it quietly disables the setting.
        """
        declared = _config_names() | COMPOSE_ONLY_VARS | THIRD_PARTY_VARS
        unknown = _interpolated_names(path) - declared
        assert not unknown, f"{path.name} references undeclared variables: {sorted(unknown)}"

    def test_no_credential_is_baked_into_a_deployment_file(self) -> None:
        """A published repo must not carry a usable token."""
        for path in (*ALL_COMPOSE, ENV_EXAMPLE):
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue  # a commented template carries no value
                match = re.match(r"PICTOR_(?:AUTH_TOKEN|URL_SECRET)\s*[:=]\s*(.*)", stripped)
                if not match:
                    continue
                value = match.group(1).strip().strip('"').strip("'")
                assert value == "" or value.startswith("${"), (
                    f"{path.name} appears to contain a literal secret: {value!r}"
                )


class TestInertAllowListEntries:
    """A pattern that can never match is protection someone thinks they have.

    Host and Origin take different shapes - `host[:port]` against
    `scheme://host[:port]` - so the two lists are easy to swap, and the result is
    silent: the entries simply never match, and the server looks like it is
    refusing a client for no reason. `PICTOR_FETCH_ALLOWED_HOSTS` is matched
    against the hostname alone, so a port there is inert for the same reason.
    """

    def test_a_clean_configuration_says_nothing(self) -> None:
        assert inert_allow_list_entries(_config()) == []

    @pytest.mark.parametrize(
        "extra",
        [(), (GPU_OVERLAY,), (ML_OVERLAY,), (BUILD_OVERLAY,)],
        ids=["cpu", "gpu", "ml", "build"],
    )
    def test_the_shipped_defaults_are_not_inert(self, extra: tuple[Path, ...]) -> None:
        """The examples are what people copy; they must not teach a dead list."""
        assert inert_allow_list_entries(load_config(_env(BASE_COMPOSE, *extra))) == []

    def test_a_scheme_in_the_host_list_is_reported(self) -> None:
        messages = inert_allow_list_entries(_config(PICTOR_ALLOWED_HOSTS="http://192.168.1.10:*"))
        assert len(messages) == 1, messages
        assert "PICTOR_ALLOWED_HOSTS" in messages[0]
        assert "PICTOR_ALLOWED_ORIGINS" in messages[0], "the message must name where the form belongs"

    def test_a_bare_host_in_the_origin_list_is_reported(self) -> None:
        messages = inert_allow_list_entries(_config(PICTOR_ALLOWED_ORIGINS="pictor-mcp:8077,pictor.example.com"))
        assert len(messages) == 2, messages
        assert all("PICTOR_ALLOWED_ORIGINS" in message for message in messages)

    def test_a_wildcard_origin_is_not_reported(self) -> None:
        """`*.example.com` and `*` are meaningful without a scheme: they are
        matched against the host part of the Origin."""
        config = _config(PICTOR_ALLOWED_ORIGINS="*,https://*.example.com")
        assert inert_allow_list_entries(config) == []

    def test_a_port_in_the_fetch_host_list_is_reported(self) -> None:
        config = _config(
            PICTOR_ALLOW_NET_FETCH="true",
            PICTOR_FETCH_ALLOWED_HOSTS="pictor-mcp:8077,cdn.example.com,[::1]:8080",
        )
        messages = inert_allow_list_entries(config)
        assert len(messages) == 2, messages
        assert all("PICTOR_FETCH_ALLOWED_HOSTS" in message for message in messages)
        assert all("PICTOR_FETCH_ALLOWED_PORTS" in message for message in messages)

    def test_a_bracketed_ipv6_host_is_not_mistaken_for_a_port(self) -> None:
        config = _config(PICTOR_ALLOW_NET_FETCH="true", PICTOR_FETCH_ALLOWED_HOSTS="[2001:db8::1]")
        assert inert_allow_list_entries(config) == []

    def test_a_base_url_without_output_serving_is_reported(self) -> None:
        """The usual reason a chat UI shows no image.

        `PICTOR_PUBLIC_BASE_URL` only ever appears inside a generated link, so
        without output serving it is a setting with no effect - and the operator
        is left wondering why the picture never arrives.
        """
        config = _config(PICTOR_PUBLIC_BASE_URL="https://pictor.example.com")
        messages = inert_allow_list_entries(config)
        assert any("PICTOR_SERVE_OUTPUTS" in message for message in messages), messages

    def test_the_warning_clears_once_serving_is_on(self) -> None:
        config = _config(
            PICTOR_PUBLIC_BASE_URL="https://pictor.example.com",
            PICTOR_SERVE_OUTPUTS="true",
            PICTOR_AUTH_TOKEN="a-sufficiently-long-token-value",
        )
        assert inert_allow_list_entries(config) == []

    def test_the_check_output_carries_the_warnings(self, tmp_path: Path) -> None:
        """`--check` is what an operator runs while working out why a client is
        refused, so the warnings have to be in it - not only in the log."""
        from pictor_mcp.server import _redacted_summary, build_context

        inputs = tmp_path / "in"
        inputs.mkdir()
        config = _config(
            PICTOR_INPUT_ROOTS=str(inputs),
            PICTOR_OUTPUT_ROOT=str(tmp_path / "out"),
            PICTOR_ALLOWED_HOSTS="http://192.168.1.10:*",
        )
        summary = _redacted_summary(config, build_context(config))
        assert summary["configurationWarnings"], summary
        assert any("PICTOR_ALLOWED_HOSTS" in message for message in summary["configurationWarnings"])


class TestProjectMetadata:
    """Metadata that ships with the artefact must describe the real project."""

    def test_package_version_matches_pyproject(self) -> None:
        import pictor_mcp

        assert pictor_mcp.__version__ == _pyproject()["version"]

    def test_the_declared_python_floor_is_a_constraint(self) -> None:
        assert _pyproject()["requires-python"].startswith(">=")

    def test_project_urls_point_at_the_real_repository(self) -> None:
        """A published artefact must not advertise a placeholder."""
        urls = _pyproject()["urls"]
        assert urls, "no project URLs declared"
        for name, value in urls.items():
            assert "TheRealChickenlegs/pictor-mcp" in value, f"{name} -> {value}"

    def test_the_ci_matrix_covers_the_image_runtime(self) -> None:
        """The interpreter the container ships must be one the matrix tests.

        These are separate declarations in separate files, and nothing kept them
        in step: a Dependabot bump moved the base image from 3.12 to 3.14 and the
        runtime the container actually uses became untested, while CI stayed
        green. The same class of gap as a duplicated registry path.
        """
        match = re.search(r"^FROM python:(\d+\.\d+)-", DOCKERFILE.read_text(), re.MULTILINE)
        assert match, "the base image does not pin a python version"
        image_python = match.group(1)

        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        matrix = re.search(r"python-version: \[([^\]]+)\]", workflow)
        assert matrix, "the CI workflow declares no python-version matrix"
        listed = {value.strip().strip('"') for value in matrix.group(1).split(",")}
        assert image_python in listed, (
            f"the image ships Python {image_python}, which the CI matrix does not test: {sorted(listed)}"
        )

    def test_the_ci_matrix_covers_the_declared_floor(self) -> None:
        """Keep requires-python and the CI matrix from disagreeing.

        They are two statements of the same promise, and a matrix that omits the
        floor version leaves the promise unverified - which is how the tomllib
        import in this very module reached a green build.
        """
        floor = _pyproject()["requires-python"]
        major, minor = re.findall(r"\d+", floor)[:2]
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        matrix = re.search(r"python-version: \[([^\]]+)\]", workflow)
        assert matrix, "the CI workflow declares no python-version matrix"
        listed = {value.strip().strip('"') for value in matrix.group(1).split(",")}
        assert f"{major}.{minor}" in listed, f"floor {major}.{minor} missing from {sorted(listed)}"


class TestDockerfile:
    def test_stage_order_is_buildable(self) -> None:
        """A stage may only inherit from an EARLIER stage.

        A forward reference would make Docker treat the name as a registry image
        and try to pull it, which is both a broken build and a supply-chain risk.
        """
        stages: list[str] = []
        for line in DOCKERFILE.read_text().splitlines():
            match = re.match(r"FROM\s+(\S+)(?:\s+AS\s+(\S+))?", line.strip(), re.IGNORECASE)
            if not match:
                continue
            base, alias = match.group(1), match.group(2)
            if base.lower() != "scratch":
                # Either a previously defined stage, or an external image that
                # must look like one (contains a tag, digest or slash).
                is_external = ":" in base or "@" in base or "/" in base
                assert is_external or base in stages, f"forward reference: FROM {base}"
            if alias:
                stages.append(alias)
        assert stages[-1] == "default", f"last stage must be the CPU alias, got {stages[-1]}"
        assert {"base", "gpu", "ml", "default"} <= set(stages)

    def test_build_overlay_target_is_a_real_stage(self) -> None:
        """The contributor path must not point at a stage that does not exist."""
        stages = set(_docker_stages())
        raw = _load(BUILD_OVERLAY)["services"][SERVICE]["build"]["target"]
        # Compose interpolates ${BUILD_TARGET:-default}; the *default* is what a
        # bare `docker compose build` uses, so it is what has to be a real stage.
        match = re.fullmatch(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}", str(raw).strip())
        assert match, f"unexpected target expression: {raw!r}"
        assert match.group(1) == "BUILD_TARGET", raw
        assert match.group(2) in stages, f"{match.group(2)!r} is not one of {sorted(stages)}"

    def test_default_stage_is_the_cpu_image(self) -> None:
        assert re.search(r"^FROM base AS default\s*$", DOCKERFILE.read_text(), re.MULTILINE)

    def test_runs_as_a_non_root_user(self) -> None:
        text = DOCKERFILE.read_text()
        assert re.search(r"^USER pictor\s*$", text, re.MULTILINE)
        assert "useradd --system --uid 10001" in text

    def test_no_credential_defaults(self) -> None:
        text = DOCKERFILE.read_text()
        assert "PICTOR_AUTH_TOKEN=" not in text
        assert "PICTOR_URL_SECRET=" not in text

    def test_healthcheck_needs_no_extra_packages(self) -> None:
        """curl is not installed; the probe must use the interpreter."""
        text = DOCKERFILE.read_text()
        assert "/healthz" in text
        assert "urllib.request" in text
        install_lines = "\n".join(
            line for line in text.splitlines() if "apt-get install" in line or line.strip().startswith("curl")
        )
        assert "curl" not in install_lines

    def test_apt_lists_are_removed_in_the_same_layer(self) -> None:
        for block in re.findall(r"RUN apt-get update(?:.|\n)*?(?=\n[A-Z]|\Z)", DOCKERFILE.read_text()):
            assert "rm -rf /var/lib/apt/lists/*" in block

    def test_torch_comes_from_a_single_index(self) -> None:
        """`--index-url`, never `--extra-index-url`.

        PyTorch documents the extra-index form, but it puts PyPI and the CUDA
        index in play together and pip then takes the highest version across
        both - which is the dependency-confusion opening. PyTorch's index
        mirrors torch's dependencies, so replacing PyPI outright resolves
        cleanly and leaves only one index to trust.
        """
        commands = _stage_commands("gpu-deps")
        assert "--index-url" in commands
        assert "--extra-index-url" not in commands

    def test_torch_is_not_pinned_to_a_fixed_version(self) -> None:
        """A hard pin plus a moving base image is a build that breaks later.

        `torch==2.4.1` stopped resolving the moment the base image moved to
        Python 3.14, because the index it named publishes no cp314 wheels. The
        default must be "whatever the index has for this interpreter".
        """
        match = re.search(r"^ARG TORCH_VERSION=(.*)$", DOCKERFILE.read_text(), re.MULTILINE)
        assert match, "TORCH_VERSION is no longer a build argument"
        assert match.group(1).strip() == "", f"TORCH_VERSION defaults to {match.group(1)!r}"

    def test_the_cuda_index_is_overridable(self) -> None:
        """An older driver needs an older CUDA build, so it must be a build arg."""
        match = re.search(r"^ARG TORCH_INDEX_URL=(\S+)$", DOCKERFILE.read_text(), re.MULTILINE)
        assert match, "TORCH_INDEX_URL is no longer a build argument"
        assert match.group(1).startswith("https://download.pytorch.org/whl/")

    def test_the_default_cuda_index_can_drive_modern_cards(self) -> None:
        """The default must be a CUDA line that includes Blackwell kernels.

        cu126 installs cleanly and is friendlier to old drivers, which is what
        made it look like the safe choice - but CUDA 12.6 predates Blackwell, so
        on an RTX 50-series card it loads, reports the GPU as available, and then
        fails every kernel launch. 12.8 is the first line with sm_120 kernels.
        """
        match = re.search(r"^ARG TORCH_INDEX_URL=(\S+)$", DOCKERFILE.read_text(), re.MULTILINE)
        assert match
        tag = match.group(1).rstrip("/").rsplit("/", 1)[-1]
        assert re.fullmatch(r"cu\d+", tag), tag
        minor = int(tag[2:])
        assert minor >= 128, f"CUDA {tag} predates Blackwell (needs cu128 or newer)"

    def test_no_comment_sits_inside_a_line_continuation(self) -> None:
        """A comment inside a continuation is stripped by Docker, but it is a
        documented gotcha and reads as if it were part of the command."""
        lines = DOCKERFILE.read_text().splitlines()
        offenders = [
            (index + 1, lines[index + 1].strip()[:60])
            for index, line in enumerate(lines[:-1])
            if line.rstrip().endswith("\\") and lines[index + 1].lstrip().startswith("#")
        ]
        assert not offenders, f"comment inside a continuation at {offenders}"

    def test_oci_source_label_is_the_real_repository(self) -> None:
        assert "TheRealChickenlegs/pictor-mcp" in DOCKERFILE.read_text()


class TestImageLayerOrder:
    """The application must be installed above every dependency layer.

    This is the difference between a code change costing seconds and costing a
    multi-gigabyte reinstall. `COPY src/` used to sit below the PyTorch install,
    so every source edit invalidated the CUDA layer and the next build fetched
    the NVIDIA wheels again - on a laptop, in CI and through the Portainer build
    API alike. These tests are the reason nobody has to remember that.
    """

    #: The variant image and the dependency stage it must inherit.
    VARIANTS: ClassVar[dict[str, str]] = {"base": "base-deps", "gpu": "gpu-deps", "ml": "ml-deps"}

    def test_variant_images_are_built_on_their_dependency_stage(self) -> None:
        stages = _docker_stages()
        for image, deps in self.VARIANTS.items():
            assert stages[image][0] == deps, f"{image} is built on {stages[image][0]}"

    def test_the_dependency_chain_is_linear_and_ordered(self) -> None:
        """gpu depends on base, ml on gpu - so the layering must be linear."""
        stages = _docker_stages()
        assert stages["base-deps"][0] == "system"
        assert stages["gpu-deps"][0] == "base-deps"
        assert stages["ml-deps"][0] == "gpu-deps"

    @pytest.mark.parametrize("alias", ["base-deps", "gpu-deps", "ml-deps"])
    def test_no_dependency_stage_copies_the_source(self, alias: str) -> None:
        """A COPY src/ here would invalidate every layer above it."""
        assert "COPY src" not in _stage_commands(alias), f"{alias} copies src/"

    @pytest.mark.parametrize("alias", ["base", "gpu", "ml"])
    def test_every_variant_copies_the_source_and_reinstalls_the_package(self, alias: str) -> None:
        commands = _stage_commands(alias)
        assert "COPY src/" in commands, f"{alias} never copies the application"
        assert "--force-reinstall --no-deps" in commands, (
            f"{alias} must replace the placeholder package unconditionally; "
            "without --force-reinstall pip is allowed to decide it is already satisfied"
        )

    @pytest.mark.parametrize("alias", ["base", "gpu", "ml"])
    def test_every_variant_verifies_the_package_it_built(self, alias: str) -> None:
        """The guard rail for the placeholder package.

        The dependency stages install an empty placeholder module, and the
        variant stage replaces it. If that replacement ever stopped happening,
        the image would build happily and die on start; asserting the import at
        build time turns that into a failed build.
        """
        commands = _stage_commands(alias)
        assert "import pictor_mcp" in commands, f"{alias} does not verify its own install"

    def test_the_placeholder_is_created_before_the_dependencies_are_installed(self) -> None:
        """`pip install .` needs a package to exist, and pyproject is the only
        dependency list. The placeholder is what lets both be true."""
        commands = _stage_commands("base-deps")
        assert "src/pictor_mcp/__init__.py" in commands
        assert "pip install" in commands

    def test_every_pip_install_is_attached_to_its_cache_mount(self) -> None:
        """A mount without PIP_CACHE_DIR (or with PIP_NO_CACHE_DIR left at the
        default of 1) is a cache that silently does nothing - a build that looks
        cache-friendly and re-downloads torch anyway."""
        installs = [
            block
            for alias in ("base-deps", "gpu-deps", "ml-deps", "base", "gpu", "ml")
            for block in _run_blocks(_stage_commands(alias))
            if "pip install" in block
        ]
        assert installs, "no pip install found in the Dockerfile"
        for block in installs:
            # --mount is a flag of the RUN instruction, not an argument of the
            # command: in the middle of the line it is handed to pip, which
            # rejects it. So it has to be the first thing after `RUN`.
            mount = re.match(r"RUN\s+--mount=type=cache,target=(\S+)\s", block)
            assert mount, f"pip install with no RUN-level cache mount: {block[:160]}"
            assert f"PIP_CACHE_DIR={mount.group(1)}" in block, block[:200]
            assert "PIP_NO_CACHE_DIR=0" in block, block[:200]

    def test_the_permission_error_names_both_remedies(self) -> None:
        """The message is the whole fix for the most common setup problem.

        It must name the variables to set and the alternative, and it must not
        hardcode the image's own uid - that is exactly the value an operator has
        overridden.
        """
        source = (ROOT / "src" / "pictor_mcp" / "server.py").read_text()
        assert "PUID" in source and "PGID" in source
        assert "chown" in source
        assert "10001" not in source


class TestDockerignore:
    def test_keeps_what_the_build_needs(self) -> None:
        text = DOCKERIGNORE.read_text()
        assert "!README.md" in text, "pyproject.toml requires README.md at build time"
        assert "!README.md" in text.split("*.md", 1)[1]

    def test_excludes_secrets_and_local_data(self) -> None:
        text = DOCKERIGNORE.read_text()
        for entry in (".env", ".git", ".venv", "input", "output", "tests"):
            assert re.search(rf"^{re.escape(entry)}$", text, re.MULTILINE), entry

    def test_does_not_need_local_secret_or_data_dirs(self) -> None:
        """The build context is pyproject + README + src, nothing else."""
        copied = re.findall(r"^COPY\s+(.+?)\s+\S+\s*$", DOCKERFILE.read_text(), re.MULTILINE)
        for entry in copied:
            for token in entry.split():
                assert token in {"pyproject.toml", "README.md", "src/", "src", "./src/"}, token

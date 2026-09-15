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

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from pictor_mcp.config import load_config

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

#: The registry path the publish workflow pushes to. Kept here so a change that
#: is not reflected in both places fails the build.
EXPECTED_IMAGE_REPO = "ghcr.io/therealcickenlegs/pictor-mcp"

_INTERPOLATION = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")

#: Variables consumed by Compose itself rather than by the server. They
#: deliberately have no PICTOR_ prefix so they cannot be confused with (or
#: accidentally documented as) server settings.
COMPOSE_ONLY_VARS = frozenset({"IMAGE_REPO", "IMAGE_TAG", "IMAGE_TAG_GPU", "IMAGE_TAG_ML", "BIND_ADDRESS"})

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

    def test_default_image_path_matches_the_publish_workflow(self) -> None:
        """The default registry path must be the one CI actually publishes to."""
        reference = _load(BASE_COMPOSE)["services"][SERVICE]["image"]
        assert reference.startswith("${IMAGE_REPO:-" + EXPECTED_IMAGE_REPO + "}"), reference

        # The publish workflow names the registry and derives the repository
        # from the checkout, so the two cannot disagree unless the registry
        # itself changes.
        workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text()
        assert "REGISTRY: ghcr.io" in workflow

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
        assert service["user"] == "10001:10001"
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
        stages = set(re.findall(r"^FROM\s+\S+\s+AS\s+(\S+)", DOCKERFILE.read_text(), re.MULTILINE))
        target = _load(BUILD_OVERLAY)["services"][SERVICE]["build"]["target"]
        assert target in stages, f"{target!r} is not one of {sorted(stages)}"

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

    def test_oci_source_label_is_the_real_repository(self) -> None:
        assert "TheRealChickenlegs/pictor-mcp" in DOCKERFILE.read_text()


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

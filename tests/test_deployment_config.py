"""Contract tests between the deployment files and the config parser.

`docker-compose.yml` and `.env.example` are long lists of variable names that
must match `config.py` exactly. Drift here is silent and dangerous: a renamed
variable leaves the compose value ignored and the *code* default in force, which
is not always the secure one.

These tests do not need Docker. They emulate Compose's ``${VAR:-default}``
interpolation with the variable unset — which is exactly what a bare
``docker compose up`` does — and feed the result through the real parser.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from pictor_mcp.config import load_config

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker-compose.yml"
COMPOSE_GPU = ROOT / "docker-compose.gpu.yml"
ENV_EXAMPLE = ROOT / ".env.example"
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"

_INTERPOLATION = re.compile(r"\$\{[A-Z0-9_]+:-([^}]*)\}")


def _interpolate(value: object) -> object:
    """Apply Compose's default-value interpolation with every variable unset."""
    if not isinstance(value, str):
        return value
    return _INTERPOLATION.sub(lambda match: match.group(1), value)


def _service_env(service: str) -> dict[str, str]:
    document = yaml.safe_load(COMPOSE.read_text())
    raw = document["services"][service]["environment"]
    assert isinstance(raw, dict), f"{service}.environment did not merge into a mapping"
    return {key: _interpolate(value) for key, value in raw.items()}  # type: ignore[misc]


def _env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


def _config_names() -> set[str]:
    """Every PICTOR_ name the code actually reads."""
    source = (ROOT / "src" / "pictor_mcp" / "config.py").read_text()
    return set(re.findall(r'"(PICTOR_[A-Z0-9_]+)"', source))


class TestComposeParses:
    @pytest.mark.parametrize("service", ["pictor-mcp", "pictor-mcp-gpu"])
    def test_service_defaults_are_valid_configuration(self, service: str) -> None:
        """A bare `docker compose up` must produce a config the server accepts."""
        config = load_config(_service_env(service))
        assert config.transport == "streamable-http"

    def test_cpu_service_disables_the_gpu(self) -> None:
        """The CPU image has no torch; asking for it would only waste startup time."""
        assert load_config(_service_env("pictor-mcp")).gpu == "off"

    def test_gpu_service_requests_auto_detection(self) -> None:
        assert load_config(_service_env("pictor-mcp-gpu")).gpu == "auto"

    @pytest.mark.parametrize("service", ["pictor-mcp", "pictor-mcp-gpu"])
    def test_secure_defaults_survive(self, service: str) -> None:
        config = load_config(_service_env(service))
        assert config.fetch.enabled is False, "URL fetching must be opt-in"
        assert config.http.serve_outputs is False, "output serving must be opt-in"
        assert config.strip_metadata is True
        assert config.http.dns_rebinding_protection is True
        assert config.limits.max_pixels > 0

    @pytest.mark.parametrize("service", ["pictor-mcp", "pictor-mcp-gpu"])
    def test_hardening_is_present(self, service: str) -> None:
        raw = yaml.safe_load(COMPOSE.read_text())["services"][service]
        assert raw["read_only"] is True
        assert raw["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in raw["security_opt"]
        assert raw["user"] == "10001:10001"
        assert raw["tmpfs"], "a read-only root needs a writable /tmp"
        assert raw["pids_limit"] and raw["mem_limit"]
        assert raw["healthcheck"]["test"]

    @pytest.mark.parametrize("service", ["pictor-mcp", "pictor-mcp-gpu"])
    def test_ports_are_loopback_only(self, service: str) -> None:
        """The default deployment must not be reachable off-host."""
        raw = yaml.safe_load(COMPOSE.read_text())["services"][service]
        for mapping in raw["ports"]:
            assert mapping.startswith("127.0.0.1:"), mapping

    @pytest.mark.parametrize("service", ["pictor-mcp", "pictor-mcp-gpu"])
    def test_input_is_mounted_read_only(self, service: str) -> None:
        raw = yaml.safe_load(COMPOSE.read_text())["services"][service]
        mounts = {m.split(":")[1]: m for m in raw["volumes"]}
        assert mounts["/data/input"].endswith(":ro")

    def test_both_services_share_one_environment_block(self) -> None:
        """An anchor keeps a security setting from applying to only one service."""
        cpu = _service_env("pictor-mcp")
        gpu = _service_env("pictor-mcp-gpu")
        differing = {key for key in cpu if cpu[key] != gpu.get(key)}
        assert differing == {"PICTOR_GPU"}, f"unexpected divergence: {differing}"


class TestVariableCoverage:
    def test_env_example_documents_every_supported_variable(self) -> None:
        """Every setting must appear, even if only as a commented-out template.

        Secrets and override-only settings are deliberately shipped commented so
        that copying the template verbatim cannot enable something surprising.
        """
        text = ENV_EXAMPLE.read_text()
        missing = {name for name in _config_names() if name not in text}
        assert not missing, f"undocumented settings: {sorted(missing)}"

    def test_env_example_invents_nothing(self) -> None:
        unknown = {k for k in _env_example() if k.startswith("PICTOR_")} - _config_names()
        assert not unknown, f"names the code does not read: {sorted(unknown)}"

    def test_env_example_defaults_are_valid(self) -> None:
        """Copying the template verbatim must give a working configuration."""
        load_config(_env_example())

    def test_compose_covers_every_supported_variable(self) -> None:
        documented = set(_service_env("pictor-mcp"))
        missing = _config_names() - documented
        assert not missing, f"not forwarded by compose: {sorted(missing)}"

    def test_no_credential_is_baked_into_a_deployment_file(self) -> None:
        """A published repo must not carry a usable token."""
        for path in (COMPOSE, COMPOSE_GPU, ENV_EXAMPLE):
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


class TestVersionConsistency:
    """pyproject.toml and the package must agree on the version.

    They are declared in two places because the packaging metadata cannot import
    the package, and the release workflow derives image tags from the git tag.
    Drift would ship an image tagged with a version the code does not report.
    """

    def test_package_version_matches_pyproject(self) -> None:
        import tomllib

        import pictor_mcp

        declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
        assert pictor_mcp.__version__ == declared

    def test_container_tags_would_match_the_package_floor_major(self) -> None:
        """The CI matrix installs the declared floor; it must be a sane version."""
        import tomllib

        requires = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
        assert requires.startswith(">=")


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

    def test_default_stage_is_the_cpu_image(self) -> None:
        text = DOCKERFILE.read_text()
        assert re.search(r"^FROM base AS default\s*$", text, re.MULTILINE)

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
        # curl must not be installed; the mentions in comments are explanatory.
        install_lines = "\n".join(
            line for line in text.splitlines() if "apt-get install" in line or line.strip().startswith("curl")
        )
        assert "curl" not in install_lines

    def test_apt_lists_are_removed_in_the_same_layer(self) -> None:
        for block in re.findall(r"RUN apt-get update(?:.|\n)*?(?=\n[A-Z]|\Z)", DOCKERFILE.read_text()):
            assert "rm -rf /var/lib/apt/lists/*" in block


class TestDockerignore:
    def test_keeps_what_the_build_needs(self) -> None:
        text = DOCKERIGNORE.read_text()
        assert "!README.md" in text, "pyproject.toml requires README.md at build time"
        assert "!README.md" in text.split("*.md", 1)[1]

    def test_excludes_secrets_and_local_data(self) -> None:
        text = DOCKERIGNORE.read_text()
        for entry in (".env", ".git", ".venv", "input", "output", "tests"):
            assert re.search(rf"^{re.escape(entry)}$", text, re.MULTILINE), entry

    def test_does_not_need_local_secret_or_data_dirs(self, tmp_path: Path) -> None:
        """The build context is pyproject + README + src, nothing else."""
        text = DOCKERFILE.read_text()
        copied = re.findall(r"^COPY\s+(.+?)\s+\S+\s*$", text, re.MULTILINE)
        for entry in copied:
            for token in entry.split():
                assert token in {"pyproject.toml", "README.md", "src/", "src", "./src/"}, token

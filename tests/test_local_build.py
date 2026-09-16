"""Contract tests for the local-build and Portainer deployment path.

The point of this path is that an image is built where it runs, so nothing is
pulled. That only holds while several files agree with each other and with the
Dockerfile: the CI pipeline's `BUILD_TARGET`, the scripts' tags, the compose
overlay's variables, and the Paths the Portainer build actually uploads.

None of it needs Docker - every script is exercised through `--print-command`,
which exits before running anything - so these run everywhere the suite runs.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
GITLAB_CI = ROOT / ".gitlab-ci.yml"
MAKEFILE = ROOT / "Makefile"
DOCKERFILE = ROOT / "Dockerfile"
BUILD_OVERLAY = ROOT / "docker-compose.build.yml"
ENV_EXAMPLE = ROOT / ".env.example"
README = ROOT / "README.md"
PORTAINER_DOC = ROOT / "docs" / "portainer.md"

BUILD_SCRIPT = ROOT / "scripts" / "build_image.sh"
PORTAINER_BUILD = ROOT / "scripts" / "portainer_build.sh"
PORTAINER_DEPLOY = ROOT / "scripts" / "portainer_deploy.sh"
SCRIPTS = (BUILD_SCRIPT, PORTAINER_BUILD, PORTAINER_DEPLOY)

#: The variants the build scripts accept, and the tag each one lands under. The
#: tag matters as much as the target: a CUDA build under the CPU tag is a
#: deployment that silently loses its GPU.
VARIANTS = {
    "default": "local",
    "gpu": "local-gpu",
    "ml": "local-ml",
}


def _docker_stages() -> set[str]:
    return set(re.findall(r"^FROM\s+\S+\s+AS\s+(\S+)\s*$", DOCKERFILE.read_text(), re.MULTILINE))


def _copied_paths() -> set[str]:
    """Every path the Dockerfile COPYs, which is everything the build needs."""
    copied = re.findall(r"^COPY\s+(.+?)\s+\S+\s*$", DOCKERFILE.read_text(), re.MULTILINE)
    return {token for entry in copied for token in entry.split()}


def _run(script: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a script with a minimal environment, so a stray host variable that
    happens to be set cannot make a test pass that would fail on a clean host."""
    clean = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp")}
    clean.update(env or {})
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True,
        text=True,
        env=clean,
        cwd=ROOT,
        timeout=30,
    )


class TestShellScripts:
    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_the_script_parses(self, script: Path) -> None:
        if not shutil.which("bash"):  # pragma: no cover - bash is everywhere CI runs
            pytest.skip("bash is not installed")
        assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_the_script_is_executable(self, script: Path) -> None:
        """Documented as `scripts/x.sh`, so it must not need `bash` in front."""
        assert os.access(script, os.X_OK), f"{script.name} is not marked executable"

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_help_works_without_any_configuration(self, script: Path) -> None:
        result = _run(script, "--help")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip(), "no usage text"

    def test_the_build_script_rejects_an_unknown_choice(self) -> None:
        """A typo must not silently build the wrong variant."""
        assert _run(BUILD_SCRIPT, "--target", "cuda").returncode != 0
        assert _run(BUILD_SCRIPT, "--nonsense").returncode != 0

    def test_the_portainer_scripts_require_their_credentials(self) -> None:
        """A missing URL or token is a configuration error, not a silent no-op."""
        assert _run(PORTAINER_BUILD, "--print-command").returncode != 0
        assert _run(PORTAINER_DEPLOY, "--print-command").returncode != 0

    def test_the_deploy_script_rejects_a_url_that_is_not_a_webhook(self) -> None:
        result = _run(PORTAINER_DEPLOY, "--print-command", env={"PORTAINER_WEBHOOK_URL": "https://x/api/stacks"})
        assert result.returncode != 0


class TestBuildScript:
    @pytest.mark.parametrize(("target", "tag"), sorted(VARIANTS.items()))
    def test_the_tag_follows_the_target(self, target: str, tag: str) -> None:
        """The tag is the safety interlock: a CUDA build must not land under the
        name a CPU stack is running."""
        result = _run(BUILD_SCRIPT, "--target", target, "--print-command")
        assert result.returncode == 0, result.stderr
        build_line = result.stdout.splitlines()[0]
        assert f"--target {target}" in build_line, build_line
        assert f"pictor-mcp:{tag}" in build_line, build_line

    def test_the_cuda_build_arguments_are_passed_through(self) -> None:
        """A driver too old for the default CUDA line is the documented reason
        these exist, so they have to survive the wrapper."""
        result = _run(
            BUILD_SCRIPT,
            "--target",
            "gpu",
            "--print-command",
            env={"TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu126", "TORCH_VERSION": "2.6.0"},
        )
        assert "--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126" in result.stdout
        assert "--build-arg TORCH_VERSION=2.6.0" in result.stdout

    def test_the_build_is_verified_before_it_is_deployed(self) -> None:
        """The smoke test is what catches an application layer that did not
        replace the dependency stage's placeholder package."""
        result = _run(BUILD_SCRIPT, "--print-command")
        lines = result.stdout.splitlines()
        assert len(lines) == 2, lines
        assert "run --rm --network none" in lines[1], lines[1]
        assert lines[1].rstrip().endswith("--version"), lines[1]

    def test_the_pushed_image_is_not_also_loaded(self) -> None:
        """--push and --load are mutually exclusive; asking for both is an error
        from buildx rather than a smaller image."""
        result = _run(BUILD_SCRIPT, "--push", "--print-command")
        assert "--push" in result.stdout
        assert "--load" not in result.stdout


class TestPortainerBuildScript:
    ENV: ClassVar[dict[str, str]] = {
        "PORTAINER_URL": "https://portainer.internal:9443",
        "PORTAINER_API_TOKEN": "token",
    }

    def test_it_calls_the_docker_build_proxy(self) -> None:
        """Portainer proxies the Docker API verbatim, which is why the build
        target can be passed at all - its UI has no field for it."""
        result = _run(PORTAINER_BUILD, "--target", "gpu", "--print-command", env=self.ENV)
        assert result.returncode == 0, result.stderr
        assert (
            "https://portainer.internal:9443/api/endpoints/1/docker/build?t=pictor-mcp:local-gpu&target=gpu"
            in result.stdout
        )
        assert "X-API-Key: token" in result.stdout
        assert "--data-binary @-" in result.stdout

    def test_it_only_uploads_what_the_build_reads(self) -> None:
        """The context crosses the network, so this is a secrecy boundary as well
        as a size one: .env and output/ must not be able to travel with it."""
        result = _run(PORTAINER_BUILD, "--print-command", env=self.ENV)
        body = next(line for line in result.stdout.splitlines() if line.startswith("body:"))
        match = re.search(r"--exclude=\*\.py\[co\]\s+(.*)$", body)
        assert match, body
        uploaded = set(match.group(1).split())
        assert "Dockerfile" in uploaded, "the daemon resolves the Dockerfile inside the context"
        # The Dockerfile writes `src/`; tar is given the same directory without
        # the trailing slash.
        missing = {path.rstrip("/") for path in _copied_paths()} - uploaded
        assert not missing, f"the build context is missing {sorted(missing)}"
        for forbidden in (".env", ".git", "output", "input", ".venv", "tests"):
            assert forbidden not in uploaded, f"{forbidden} would be uploaded"

    def test_the_endpoint_and_insecure_switch_are_configurable(self) -> None:
        result = _run(
            PORTAINER_BUILD,
            "--print-command",
            "--insecure",
            env={**self.ENV, "PORTAINER_ENDPOINT_ID": "7"},
        )
        assert "/api/endpoints/7/docker/build" in result.stdout
        assert result.stdout.splitlines()[0].rstrip().endswith("-k")

    def test_cuda_arguments_are_encoded_into_the_query(self) -> None:
        """They are a JSON buildargs parameter on a request whose body is a tar,
        so --data-urlencode is not available and the encoding is manual."""
        result = _run(
            PORTAINER_BUILD,
            "--print-command",
            env={**self.ENV, "TORCH_VERSION": "2.9.0"},
        )
        assert "buildargs=%7B%22TORCH_VERSION%22%3A%222.9.0%22%7D" in result.stdout


class TestPortainerDeployScript:
    URL = "https://portainer.internal:9443/api/stacks/webhooks/abc-123"

    def test_a_redeploy_never_pulls_by_default(self) -> None:
        """The image was built on this host; pulling it would fail, and pulling a
        published one of the same name would defeat the whole arrangement."""
        result = _run(PORTAINER_DEPLOY, "--print-command", env={"PORTAINER_WEBHOOK_URL": self.URL})
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith(f"{self.URL}?pullimage=false")

    def test_pulling_can_be_requested_explicitly(self) -> None:
        result = _run(PORTAINER_DEPLOY, "--print-command", "--pull", env={"PORTAINER_WEBHOOK_URL": self.URL})
        assert "pullimage" not in result.stdout

    def test_a_tag_and_an_existing_query_are_combined_correctly(self) -> None:
        """A second '?' would silently drop both parameters."""
        result = _run(
            PORTAINER_DEPLOY,
            "--print-command",
            "--tag",
            "1.0.0",
            env={"PORTAINER_WEBHOOK_URL": f"{self.URL}?existing=1"},
        )
        assert "?existing=1&pullimage=false&tag=1.0.0" in result.stdout
        assert result.stdout.count("?") == 1


class TestGitLabPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self) -> dict:
        return yaml.safe_load(GITLAB_CI.read_text())

    def test_every_job_declares_a_known_stage(self, pipeline: dict) -> None:
        stages = pipeline["stages"]
        for name, job in pipeline.items():
            if name in {"stages", "variables", "default", "workflow"} or name.startswith("."):
                continue
            assert job["stage"] in stages, f"{name} uses undeclared stage {job['stage']!r}"

    def test_build_jobs_target_real_dockerfile_stages(self, pipeline: dict) -> None:
        """A renamed or mistyped stage would fail the pipeline rather than the
        deployment, but the message would be a Docker error nowhere near here."""
        stages = _docker_stages()
        build_jobs = {
            name: job
            for name, job in pipeline.items()
            if isinstance(job, dict) and job.get("stage") == "build" and name.startswith("build:")
        }
        assert build_jobs, "no build jobs"
        for name, job in build_jobs.items():
            target = (job.get("variables") or {}).get("BUILD_TARGET")
            if target is None:  # build:push takes its target from the environment
                continue
            assert target in stages, f"{name} targets {target!r}, not one of {sorted(stages)}"

    def test_build_jobs_select_the_target_through_the_scripts(self, pipeline: dict) -> None:
        for name, job in pipeline.items():
            if not (isinstance(job, dict) and name.startswith("build:")):
                continue
            script = " ".join(job["script"])
            assert "scripts/build_image.sh" in script, f"{name} does not use the build script"
            assert "--target" in script, f"{name} does not choose a stage"

    def test_the_deploy_job_uses_the_script_that_disables_pulling(self, pipeline: dict) -> None:
        """A hand-written curl here would be one edit away from dropping
        pullimage=false, which is the parameter that makes this whole path work."""
        deploy = pipeline["deploy:stack"]
        assert any("scripts/portainer_deploy.sh" in line for line in deploy["script"])
        assert all("curl" not in line for line in deploy["script"])

    def test_the_pipeline_runs_the_checks_the_other_ci_does(self, pipeline: dict) -> None:
        """Two CI systems for one repository is one place for the floor check to
        go missing."""
        text = GITLAB_CI.read_text()
        assert "vermin -t=3.10-" in text, "the declared Python floor is not enforced"
        assert "pytest" in text, "the test suite is not run"
        assert "ruff check" in text
        assert "bash -n" in text, "the shell scripts are never parsed"

    def test_no_job_runs_docker_privileged(self, pipeline: dict) -> None:
        """The build reuses the daemon the deployment already trusts. Handing a
        job its own privileged daemon is a different security claim, so it must
        be a deliberate edit rather than a default nobody noticed."""
        assert "privileged" not in GITLAB_CI.read_text()


class TestMakefile:
    @pytest.fixture(scope="class")
    def targets(self) -> dict[str, str]:
        bodies: dict[str, list[str]] = {}
        current: str | None = None
        for line in MAKEFILE.read_text().splitlines():
            match = re.match(r"^([a-zA-Z][a-zA-Z0-9_-]*):", line)
            if match and not line.startswith("\t"):
                current = match.group(1)
                bodies[current] = []
            elif current is not None:
                bodies[current].append(line)
        return {name: "\n".join(lines) for name, lines in bodies.items()}

    def test_this_is_not_a_second_implementation(self, targets: dict) -> None:
        """Every target is a wrapper; logic belongs in the scripts the pipeline
        also calls, or the two drift."""
        for name in ("build", "build-gpu", "build-ml"):
            assert "scripts/build_image.sh" in targets[name], name
        assert "scripts/portainer_build.sh" in targets["portainer-build"]
        assert "scripts/portainer_deploy.sh" in targets["deploy"]

    def test_the_built_tag_matches_the_variant(self, targets: dict) -> None:
        """`make up-gpu` must run the image `make build-gpu` produced."""
        assert "--target gpu" in targets["build-gpu"]
        assert "LOCAL_IMAGE_TAG=local-gpu" in targets["up-gpu"]
        assert "--target ml" in targets["build-ml"]
        assert "LOCAL_IMAGE_TAG=local-ml" in targets["up-ml"]
        assert "--target default" in targets["build"]
        # Not `local-gpu`: the CPU stack must not pick up a CUDA image.
        assert re.search(r"LOCAL_IMAGE_TAG=local(?![\w-])", targets["up"])

    def test_phony_names_all_exist(self, targets: dict) -> None:
        declared = set(re.findall(r"^\.PHONY:\s*(.+)$", MAKEFILE.read_text(), re.MULTILINE))
        names = {name for entry in declared for name in entry.split()}
        missing = names - set(targets)
        assert not missing, f".PHONY names with no target: {sorted(missing)}"

    @pytest.mark.parametrize("target", ["help", "build", "build-gpu", "build-ml", "up", "up-gpu", "up-ml", "check"])
    def test_make_can_parse_and_expand_every_target(self, target: str) -> None:
        """`make -n` parses the whole file and expands one recipe, which is the
        cheapest way to catch a space-indented recipe or a bad variable."""
        if not shutil.which("make"):  # pragma: no cover - make is everywhere CI runs
            pytest.skip("make is not installed")
        result = subprocess.run(["make", "-n", target], capture_output=True, text=True, cwd=ROOT, timeout=30)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip(), f"make -n {target} expanded to nothing"


class TestDocumentedArrangement:
    def test_the_portainer_doc_exists_and_is_linked(self) -> None:
        assert PORTAINER_DOC.is_file()
        assert "docs/portainer.md" in README.read_text()

    def test_the_scripts_and_the_compose_overlay_agree_on_the_tag(self) -> None:
        """The overlay builds `pictor-mcp:${LOCAL_IMAGE_TAG}`; the scripts must
        default to the same tags or `make build-gpu` and `make up-gpu` disagree."""
        overlay = BUILD_OVERLAY.read_text()
        assert "pictor-mcp:${LOCAL_IMAGE_TAG:-local}" in overlay
        assert "${BUILD_TARGET:-default}" in overlay
        script = BUILD_SCRIPT.read_text()
        for tag in VARIANTS.values():
            assert f'"{tag}"' in script, f"{tag} is not a tag the build script knows"

    def test_the_stack_side_variables_are_documented(self) -> None:
        """The deploy half reads IMAGE_*, the build half reads LOCAL_IMAGE_TAG;
        both appear in .env.example so neither is guesswork."""
        text = ENV_EXAMPLE.read_text()
        for name in ("BUILD_TARGET", "LOCAL_IMAGE_TAG", "IMAGE_TAG_GPU", "IMAGE_TAG_ML"):
            assert f"{name}=" in text, name
        assert "docs/portainer.md" in text

    def test_the_doc_does_not_claim_portainer_builds_on_push(self) -> None:
        """The one thing a reader is most likely to assume, and the one thing
        Portainer's own documentation says it does not do."""
        doc = PORTAINER_DOC.read_text()
        assert "does not build" in doc or "not build one" in doc or "not fully implemented" in doc
        assert "pullimage=false" in doc

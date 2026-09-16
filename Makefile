# pictor-mcp - the handful of commands worth not retyping.
#
# Everything here is a thin wrapper over docker, curl and pytest; `make` is a
# convenience, not a second implementation, and every target is a command you
# could have typed. Run `make` with no arguments for the list.
#
# The build targets exist because the images are large. Each one builds the
# Dockerfile stage that matches the tag it uses, keeps the multi-gigabyte
# dependency layers below `COPY src/` (see the Dockerfile), and verifies the
# result by starting it with no network before you deploy it.
#
# Inside a virtualenv, `make check PYTHON=.venv/bin/python` is the usual shape;
# every other target only needs docker and curl.

SHELL := /bin/bash
.DEFAULT_GOAL := help

DOCKER ?= docker
COMPOSE ?= docker compose
PYTHON ?= python3
BUILD_TARGET ?= default

# Values for `make portainer-build`, which builds on the Portainer host instead
# of here. See docs/portainer.md.
PORTAINER_ENDPOINT_ID ?= 1

.PHONY: help
help: ## Show this help
	@printf 'pictor-mcp targets:\n\n'
	@grep -hE '^[a-zA-Z][a-zA-Z0-9_-]*:.*?## ' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@printf '\nVariables: BUILD_TARGET=%s COMPOSE="%s"\n' '$(BUILD_TARGET)' '$(COMPOSE)'

# --------------------------------------------------------------------- builds
.PHONY: build build-gpu build-ml
build: ## Build the CPU image into the local daemon (pictor-mcp:local)
	scripts/build_image.sh --target default

build-gpu: ## Build the CUDA image locally (pictor-mcp:local-gpu)
	scripts/build_image.sh --target gpu

build-ml: ## Build the CUDA + background-removal image locally (pictor-mcp:local-ml)
	scripts/build_image.sh --target ml

.PHONY: portainer-build
portainer-build: ## Build on the Portainer host: PORTAINER_URL=... PORTAINER_API_TOKEN=... make portainer-build BUILD_TARGET=gpu
	PORTAINER_ENDPOINT_ID='$(PORTAINER_ENDPOINT_ID)' \
		scripts/portainer_build.sh --target '$(BUILD_TARGET)'

.PHONY: deploy
deploy: ## Trigger the Portainer stack webhook without pulling: PORTAINER_WEBHOOK_URL=... make deploy
	scripts/portainer_deploy.sh

# ----------------------------------------------------------------- running it
.PHONY: up up-gpu up-ml down logs
up: ## Build the CPU image and (re)start the stack from it
	BUILD_TARGET=default LOCAL_IMAGE_TAG=local \
		$(COMPOSE) -f docker-compose.yml -f docker-compose.build.yml up -d --build

up-gpu: ## Build the CUDA image and (re)start the stack from it
	BUILD_TARGET=gpu LOCAL_IMAGE_TAG=local-gpu \
		$(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.build.yml up -d --build

up-ml: ## Build the ML image and (re)start the stack from it
	BUILD_TARGET=ml LOCAL_IMAGE_TAG=local-ml \
		$(COMPOSE) -f docker-compose.yml -f docker-compose.ml.yml -f docker-compose.build.yml up -d --build

down: ## Stop the stack
	$(COMPOSE) -f docker-compose.yml down

logs: ## Follow the container log
	$(COMPOSE) -f docker-compose.yml logs -f

# ------------------------------------------------------------------- checking
.PHONY: check
check: ## Lint, format, version-floor and test (what CI runs)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m vermin -t=3.10- --no-tips --violations src tests scripts
	$(PYTHON) -m pytest -q
.PHONY: compose-check
compose-check: ## Validate the compose files and the deployment contract
	$(COMPOSE) -f docker-compose.yml config -q
	$(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml config -q
	$(COMPOSE) -f docker-compose.yml -f docker-compose.ml.yml config -q
	$(COMPOSE) -f docker-compose.yml -f docker-compose.build.yml config -q
	$(PYTHON) -m pytest tests/test_deployment_config.py tests/test_local_build.py -q

.PHONY: smoke
smoke: ## Start the locally built CPU image with no network and ask for its version
	$(DOCKER) run --rm --network none pictor-mcp:local --version

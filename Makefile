# FactoryVision developer task runner.
#
# Override command variables when a tool is not on PATH, for example:
#   make PYTHON=.venv/Scripts/python.exe test

.DEFAULT_GOAL := help

PYTHON ?= python
KEDRO ?= kedro
UVICORN ?= uvicorn
DOCKER_COMPOSE ?= docker compose
KUBECTL ?= kubectl
K8S_DIR ?= k8s

.PHONY: help train serve test docker docker-down k8s k8s-dry-run

help: ## Show the available developer commands
	@echo "FactoryVision commands:"
	@echo "  make train       Run the Kedro training and evaluation pipeline"
	@echo "  make serve       Start the FastAPI inference service"
	@echo "  make test        Run the automated test suite"
	@echo "  make docker      Build and start the Docker Compose platform"
	@echo "  make docker-down Stop the Docker Compose platform"
	@echo "  make k8s         Apply the Kubernetes manifests"
	@echo "  make k8s-dry-run Validate Kubernetes manifests without a cluster"

train: ## Run the Kedro ingestion, preprocessing, training, and evaluation pipeline
	$(KEDRO) run

serve: ## Start the FastAPI service on localhost:8000
	$(UVICORN) factoryvision.api.main:app --host 127.0.0.1 --port 8000

test: ## Run the automated test suite
	$(PYTHON) -m pytest

docker: ## Build and start the local Docker Compose platform in the background
	$(DOCKER_COMPOSE) up --build --detach

docker-down: ## Stop the local Docker Compose platform
	$(DOCKER_COMPOSE) down

k8s: ## Apply the Kubernetes manifests through Kustomize
	$(KUBECTL) apply -k $(K8S_DIR)

k8s-dry-run: ## Validate Kubernetes manifests without changing a cluster
	$(KUBECTL) apply --dry-run=client -k $(K8S_DIR)

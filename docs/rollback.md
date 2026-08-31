# Rollback runbook

This runbook describes how to restore a previously known-good FactoryVision
container and model when a new release causes health-check failures, elevated
errors, higher latency, or incorrect predictions.

## What must be rolled back together

The API container and the ONNX model are separate artifacts in this project:

- the container is published to GitHub Container Registry (GHCR);
- the generated model is mounted from the model directory and is not committed
  to Git or copied into the Docker image.

Always identify a matching container commit-SHA tag and model artifact before
starting the rollback. Rolling back only one of them can produce an
inconsistent release.

## Choose the known-good version

Use a commit SHA from the last successful GitHub Actions run or from a release
that was previously validated. Prefer the immutable SHA tag over `latest`:

```text
ghcr.io/ekrov/factoryvision-mlops:<known-good-commit-sha>
```

The `latest` tag is convenient for development but can move when a new push
to `main` is published. A commit-SHA tag stays tied to one source revision.

If the GHCR package is private, authenticate with an account or token that has
package-read permission. Keep that credential in the local credential store or
cluster secret; never put it in this document, a command committed to Git, or
an image tag.

## Kubernetes rollback

### Option A: undo the latest Deployment revision

Use this when the immediately previous ReplicaSet is known to be good:

```powershell
kubectl rollout history deployment/factoryvision-api
kubectl rollout undo deployment/factoryvision-api
kubectl rollout status deployment/factoryvision-api
```

Inspect the pods and recent events after the rollout:

```powershell
kubectl get pods -l app.kubernetes.io/name=factoryvision-api
kubectl get events --sort-by=.lastTimestamp
```

### Option B: select an explicit container version

Use this when the known-good version is not the immediately previous
Deployment revision:

```powershell
kubectl set image deployment/factoryvision-api `
  api=ghcr.io/ekrov/factoryvision-mlops:<known-good-commit-sha>
kubectl rollout status deployment/factoryvision-api
```

Replace `<known-good-commit-sha>` with the exact SHA. For a private GHCR
package, the cluster must already have an appropriate `imagePullSecret`; the
secret setup is environment-specific and must not be hard-coded in this repo.

The repository's local kind setup uses the locally loaded image
`factoryvision-api:latest` rather than GHCR. For that setup, build or load the
desired image into the kind node first, then set the Deployment image to that
local tag:

```powershell
kind load docker-image factoryvision-api:<known-good-tag> --name factoryvision
kubectl set image deployment/factoryvision-api `
  api=factoryvision-api:<known-good-tag>
kubectl rollout status deployment/factoryvision-api
```

### Restore the matching model

Before or during the rollout, restore the ONNX artifact that belongs to the
selected container version at the path mounted by the Deployment:

```text
/mnt/factoryvision-models/factoryvision-segmentation.onnx
```

The current Kubernetes manifest mounts that host path at
`/app/artifacts/models` inside the pod. After replacing the model file, restart
the Deployment if the application does not reload it automatically:

```powershell
kubectl rollout restart deployment/factoryvision-api
kubectl rollout status deployment/factoryvision-api
```

## Docker Compose rollback

The current Compose stack builds the API from the checked-out source instead
of pulling a registry image. To reproduce an older Compose release, use a
separate worktree at the known-good commit so the current checkout remains
untouched:

```powershell
git worktree add ..\factoryvision-rollback <known-good-commit-sha>
Set-Location ..\factoryvision-rollback
docker compose up -d --build api
docker compose ps
```

Restore the matching model artifact in the Compose model location described in
[`docs/configuration.md`](configuration.md), then verify the API. Remove the
temporary worktree only after the rollback is no longer needed:

```powershell
Set-Location ..\Study_Github
git worktree remove ..\factoryvision-rollback
```

Do not use `git reset --hard` on the active checkout as a rollback procedure;
it can discard uncommitted learning work.

## Verify the rollback

Run the same checks used for a normal deployment:

```powershell
kubectl port-forward service/factoryvision-api 8000:8000
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/model-info
```

Then send one known non-defect image and one known defect image through
`/predict`. Confirm that:

1. `/health` reports both `model_loaded` and `storage_ready` as healthy;
2. `/model-info` shows the expected runtime and input shape;
3. predictions are persisted successfully;
4. Prometheus shows the expected request/error counts and latency; and
5. Grafana returns to the previous error-rate and latency range.

If any check fails, inspect pod logs and events, restore the last confirmed
version, and keep the failed version identified for investigation. Record the
container SHA, model version, rollback reason, and verification result in the
release notes or incident record.

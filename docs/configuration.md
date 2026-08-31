# Configuration and secrets

FactoryVision reads runtime configuration from environment variables. The
repository includes [`.env.example`](../.env.example) as a safe template for
local Docker Compose use. Copy it to `.env` and edit the values locally:

```powershell
Copy-Item .env.example .env
docker compose --env-file .env up -d
```

The real `.env` file is ignored by Git. The example file contains only local
demo values and is safe to publish; replace those values before using a real
database or monitoring system.

## Application variables

| Variable | Purpose | Example or default |
| --- | --- | --- |
| `FACTORYVISION_DATABASE_URL` | SQLAlchemy connection string used for prediction storage | Compose: `postgresql+psycopg://...@postgres:5432/factoryvision` |
| `FACTORYVISION_ONNX_MODEL` | Path to the ONNX model inside the running environment | `/app/artifacts/models/factoryvision-segmentation.onnx` |
| `FACTORYVISION_MODEL_NAME` | Logical model name stored in predictions and metrics | `factoryvision-segmentation` |
| `FACTORYVISION_MODEL_ALIAS` | Serving alias, such as `candidate` or `production` | `candidate` |
| `FACTORYVISION_THRESHOLD` | Probability threshold used to turn pixels into defect/not-defect | `0.5` |
| `FACTORYVISION_MAX_UPLOAD_BYTES` | Maximum accepted upload size | `10000000` |
| `FACTORYVISION_BATCH_IMAGE_DIR` | Directory scanned by batch inference | `/app/data/batch/incoming` in Compose |
| `FACTORYVISION_BATCH_ARTIFACT_DIR` | Directory for batch hand-off and summary artifacts | `/app/artifacts/batch` in Compose |
| `FACTORYVISION_BATCH_MAX_IMAGES` | Optional limit for one batch run | Empty means no limit |

For direct local API execution, use host paths and `localhost` in the database
URL. For Docker Compose, use container paths and the service name `postgres`.
The same distinction applies to Kubernetes, where paths must exist inside the
pod and the database hostname must be a Kubernetes Service.

## Service configuration

| Variable | Used by | Sensitive? |
| --- | --- | --- |
| `POSTGRES_DB` | PostgreSQL container | No, but database details should still be environment-specific |
| `POSTGRES_USER` | PostgreSQL and clients | No for the local demo |
| `POSTGRES_PASSWORD` | PostgreSQL initialization | Yes outside local development |
| `MLFLOW_BACKEND_STORE_URI` | MLflow metadata database | Contains credentials when using authenticated SQL |
| `GRAFANA_ADMIN_USER` | Grafana login | No for the username |
| `GRAFANA_ADMIN_PASSWORD` | Grafana login | Yes outside local development |
| `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS` | Local Airflow demo login | Contains a demo credential |
| `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE` | Airflow password-file location | No; the generated file is runtime state |

Compose reads `.env` automatically; `docker-compose.yml` also supplies local
defaults when a variable is absent. Use `docker compose config` to inspect the
resolved configuration, but do not paste resolved output into a public issue
or commit it because it may contain passwords.

## Where secrets belong

- Local development: `.env`, kept outside Git.
- GitHub Actions: the automatically provided `GITHUB_TOKEN` is used for GHCR;
  do not create a password line in `.env` for it.
- Kubernetes: use a Kubernetes `Secret` for database URLs, passwords, and
  registry credentials. Keep ordinary model and threshold settings in a
  `ConfigMap`. The current local manifests use demo values and document where
  a Secret should replace them.
- Cloud deployment: use the cloud provider's secret store or managed
  environment-variable mechanism rather than committing credentials.

Never commit `.env`, access tokens, database passwords, private keys, or
resolved deployment output. If a credential is exposed, rotate it rather than
only deleting it from the working tree.

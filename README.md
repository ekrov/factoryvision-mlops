# FactoryVision

## Production-Grade Surface Defect Segmentation & MLOps Platform

FactoryVision is an end-to-end industrial visual-inspection system for detecting and segmenting manufacturing surface defects. The project covers the complete machine-learning lifecycle:

**data versioning -> preprocessing -> GPU training -> evaluation -> experiment tracking -> model registry -> optimized serving -> batch inference -> deployment -> monitoring**

The target outcome is a reproducible, production-oriented computer-vision platform rather than an isolated research notebook.

> This README describes the complete target system. Metrics, benchmarks, deployment endpoints, and screenshots will be added as they are produced.

## Project goals

- Train a PyTorch segmentation model for industrial surface defects.
- Version datasets and models for reproducible experiments.
- Expose real-time and batch inference through production-style services.
- Persist prediction records for SQL and offline analytics.
- Containerize and deploy the system locally and to cloud infrastructure.
- Monitor service health, inference performance, and model/business metrics.
- Demonstrate software engineering practices including testing, CI/CD, orchestration, and observability.

## Problem and dataset

FactoryVision targets pixel-level industrial defect inspection using the [KolektorSDD2 dataset](https://www.vicos.si/resources/kolektorsdd2/). The system will produce:

- a predicted defect mask;
- a defect/no-defect score;
- a derived bounding box;
- segmentation and defect-level evaluation metrics.

The initial model will be a PyTorch U-Net or DeepLabV3 implementation. The project prioritizes reproducibility and production integration over adding multiple deep-learning frameworks or unnecessarily novel architectures.

## Dataset examples

These are raw dataset inputs before model inference. The left image shows representative defect types from the official dataset overview; the right image is a defect-free training sample. The model will learn to distinguish these cases and, for defective inputs, produce a pixel-level mask.

<table>
  <tr>
    <th>Defect examples</th>
    <th>Non-defect example</th>
  </tr>
  <tr>
    <td>
      <img src="https://www.vicos.si/resources/kolektorsdd2/images/kolektor-sdd2-types.png" alt="Representative KolektorSDD2 surface defect examples" width="420" />
    </td>
    <td>
      <img src="assets/dataset/non_defect_sample.jpg" alt="KolektorSDD2 defect-free surface sample" width="240" />
    </td>
  </tr>
  <tr>
    <td>Scratches, spots, and other surface imperfections.</td>
    <td>A surface image labeled without a visible defect.</td>
  </tr>
</table>

The defect overview image is provided by [ViCoS Lab](https://www.vicos.si/resources/kolektorsdd2/). The non-defect sample is from a [KolektorSDD2 dataset mirror](https://huggingface.co/datasets/sizhkhy/kolektor_sdd2) and is stored locally so the example renders reliably on GitHub. Dataset usage remains subject to the original [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Architecture

```mermaid
flowchart LR
    A["KolektorSDD2 images + masks"] --> B["DVC dataset version"]
    B --> C["Kedro preprocessing pipeline"]
    C --> D["PyTorch training"]
    D --> E["MLflow experiments + model registry"]
    E --> F["ONNX export"]
    F --> G["FastAPI inference service"]
    G --> H["PostgreSQL prediction logs"]
    G --> I["Prometheus metrics"]
    I --> J["Grafana dashboard"]
    G --> K["Docker"]
    K --> L["Kubernetes / Azure deployment"]
    M["Airflow"] --> C
    M --> G
    N["GitHub Actions"] --> K
    O["PySpark offline analytics"] --> H
```

## Technology stack

| Area | Technologies and concepts |
| --- | --- |
| Computer vision / ML | Python, PyTorch, OpenCV, GPU training |
| Model architecture | U-Net, DeepLabV3, semantic segmentation, defect detection |
| Data engineering | Kedro, DVC, Pandas, PySpark, SQL, PostgreSQL |
| Experiment lifecycle | MLflow, experiment tracking, model registry, reproducibility |
| Model serving | FastAPI, ONNX, ONNX Runtime, REST API |
| Batch orchestration | Apache Airflow, scheduled inference, data validation |
| Packaging and platform | Docker, Docker Compose, Kubernetes, Azure |
| CI/CD | GitHub Actions, GitHub Container Registry, automated tests |
| Observability | Prometheus, Grafana, latency, throughput, error rate, defect rate |
| Edge / systems stretch | C++, OpenCV, ONNX Runtime, inference optimization |

## Inference API

The FastAPI service will provide:

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | Accept an image and return a defect mask, probability, and bounding box |
| `GET /health` | Liveness and service health check |
| `GET /model-info` | Model version, evaluation metrics, and runtime metadata |

The service will include request validation, clear error responses, automated `pytest` tests, and a comparison between PyTorch and ONNX Runtime inference.

## Evaluation and performance

Model quality will be reported with:

- Dice score;
- Intersection over Union (IoU);
- precision and recall;
- defect-level F1 score;
- qualitative prediction overlays.

Serving performance will be benchmarked with:

- model size;
- mean inference latency;
- p50 and p95 latency;
- throughput;
- error rate.

## Observability

Prometheus metrics and Grafana dashboards will cover:

- request count and error rate;
- inference latency histograms;
- throughput;
- current model version;
- predicted defect rate;
- service and model/business health.

Prediction logs will be stored in PostgreSQL with image ID, timestamp, model version, defect score, prediction latency, and status. A small PySpark job will compute daily statistics by model version and defect status.

## Implementation roadmap

### Data, repository, and baseline model

- Set up the project structure, Python environment, and pinned dependencies.
- Download KolektorSDD2 and track it with DVC; raw data does not belong in Git.
- Build an exploratory data-analysis notebook.
- Train a first PyTorch U-Net or DeepLabV3 baseline.
- Report segmentation and defect-level metrics with prediction overlays.

### Reproducible training

- Convert ingestion, preprocessing, training, and evaluation into Kedro nodes and pipelines.
- Add configuration for paths, augmentations, seeds, model settings, and hyperparameters.
- Track parameters, metrics, artifacts, and example predictions in MLflow.
- Register the best model as `factoryvision-segmentation`.
- Run three to five comparable experiments.

### Real-time inference and ONNX optimization

- Export the best model to ONNX.
- Benchmark PyTorch against ONNX Runtime.
- Implement and test the FastAPI inference service.
- Add request validation, health checks, model metadata, and error handling.
- Optionally implement a C++/OpenCV/ONNX Runtime inference executable.

### Batch inference and data processing

- Store prediction metadata and results in PostgreSQL.
- Create an Airflow DAG for discovery, validation, batch inference, persistence, and summary generation.
- Add PySpark offline analytics for daily prediction statistics.

### Containers, Kubernetes, and monitoring

- Build a multi-stage Docker image.
- Add Docker Compose for the API, PostgreSQL, MLflow, Prometheus, and Grafana.
- Create Kubernetes Deployment, Service, ConfigMap, and resource-limit manifests.
- Run the stack locally with kind, k3d, or minikube.

### CI/CD and deployment

- Add a GitHub Actions pipeline for linting, tests, and Docker builds.
- Push successful images to GitHub Container Registry.
- Deploy the FastAPI container to Azure Container Apps or another simple Azure target.
- Document secrets, environment variables, load-test results, and rollback procedures.

### Portfolio and release readiness

- Complete the README, architecture diagram, results, screenshots, and demo.
- Add a Model Card and Data Card covering intended use, limitations, and failure modes.
- Add representative input/prediction examples.
- Record a short flow from image upload to API prediction, Grafana metrics, and MLflow run.
- Add a Makefile or task runner and create the `v1.0.0` GitHub release.
- Optionally generate a daily inspection report from structured batch metrics using an LLM downstream of the computer-vision predictions.

## Project capabilities

### Primary platform capabilities

PyTorch segmentation, DVC, Kedro, MLflow, FastAPI, SQL/PostgreSQL, Airflow, Docker, Prometheus/Grafana, GitHub Actions, and one deployment target.

### Extended capabilities

PySpark, local Kubernetes, C++ ONNX inference, edge-runtime optimization, and LLM-generated inspection reports.

A complete, documented production pipeline is more valuable than touching many tools superficially.

## Repository structure

```text
factoryvision-mlops/
|-- .github/workflows/ci.yml
|-- configs/
|-- conf/base/             # Kedro parameters and catalog configuration
|-- data/                  # DVC pointers only
|-- docker/
|-- k8s/
|-- notebooks/
|-- src/factoryvision/
|   |-- data/
|   |-- pipelines/baseline/ # Kedro nodes and pipeline definition
|   |-- training/
|   |-- inference/
|   |-- api/
|   `-- monitoring/
|-- airflow/dags/
|-- spark/
|-- tests/
|-- dvc.yaml
|-- docker-compose.yml
|-- Makefile
|-- pyproject.toml
|-- MODEL_CARD.md
`-- README.md
```

## Project commands

```bash
kedro run
make train
make serve
make test
make docker
make k8s
```

These are the intended developer entry points for the corresponding components.

## Kedro configuration

The baseline pipeline reads its experiment settings from `conf/base/parameters.yml`.
The sections are intentionally separated by responsibility:

| Section | Controls |
| --- | --- |
| `dataset` | DVC manifest path and target image dimensions |
| `augmentation` | Training-only image/mask transforms and their probabilities |
| `model` | U-Net input channels, output channels, and base feature width |
| `training` | Seed, deterministic mode, device, optimizer, loss, batch size, epochs, and output directory |
| `evaluation` | Probability threshold and qualitative examples per class |

Edit those YAML values before running `kedro run`; the Python pipeline code does not need to change for ordinary experiments. Training augmentations are applied to the image and mask together, while validation augmentations are disabled so evaluation remains comparable across runs. Machine-specific overrides can be placed in `conf/local/`.

## MLflow experiment tracking

Kedro training runs are tracked locally with MLflow. The tracking configuration is in the `tracking` section of `conf/base/parameters.yml`; the default backend is the SQLite database `artifacts/mlflow.db`, and generated artifacts remain under the ignored `artifacts/` directory.

Run the pipeline to create an experiment run:

```powershell
.venv\Scripts\kedro.exe run
```

Start the local MLflow UI in a second terminal:

```powershell
.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///artifacts/mlflow.db
```

Then open `http://127.0.0.1:5000`. Each run contains the flattened configuration parameters, per-epoch losses and metrics, the best checkpoint, the training history/configuration files, and the validation prediction overlay.

To run the controlled mini-study, use the same split, seed, and epoch budget for each configuration:

```powershell
.venv\Scripts\python.exe scripts\run_mini_study.py --limit 3 --epochs 2
```

The study compares the baseline, a lower learning rate, and a balanced BCE/Dice loss. It selects the winner by final IoU, using defect-level F1 as a tie-breaker, and writes the comparison to `artifacts/mini-study/results.csv`.

Mini-study snapshot, using two epochs on CUDA with the same seed and official split:

| Configuration | Learning rate | BCE/Dice weights | Final IoU | Final Dice | Defect F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.001 | 0.25 / 0.75 | 0.0489 | 0.0932 | 0.5029 |
| Lower learning rate | 0.0003 | 0.25 / 0.75 | 0.0155 | 0.0306 | 0.3896 |
| Balanced loss | 0.001 | 0.50 / 0.50 | 0.0211 | 0.0413 | 0.2623 |

The baseline was selected by final IoU. These are deliberately short comparison runs, so the winner is a candidate for a longer training run rather than a final quality claim.

## MLflow Model Registry

The selected mini-study winner can be registered as the versioned MLflow model `factoryvision-segmentation`:

```powershell
.venv\Scripts\python.exe scripts\register_best_model.py
```

The script reads `artifacts/mini-study/winner.json`, retrieves that run's tracked `model/best.pt` artifact, rebuilds the U-Net from the run's logged architecture parameters, and logs the complete PyTorch model with an input/output signature. MLflow then creates a model version in the local registry stored in `artifacts/mlflow.db` and assigns the version the `candidate` alias.

The registered model can be loaded by alias:

```python
import mlflow.pytorch

model = mlflow.pytorch.load_model(
    "models:/factoryvision-segmentation@candidate"
)
```

The alias identifies the currently selected candidate without putting model binaries in Git. The registration is intentionally local for now; the MLflow database and model artifacts are ignored by Git and can later be moved to a shared MLflow server or artifact store.

## Reproducibility

FactoryVision uses explicit configuration and deterministic seed handling so experiments can be repeated consistently.

### Reproduce the baseline run

Use Python 3.11 and install the project dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[training,dev]"
```

Restore the DVC-tracked dataset and verify that the reusable split manifest exists:

```powershell
.venv\Scripts\dvc.exe pull
```

```text
data/processed/splits.csv
```

The training configuration is in `conf/base/parameters.yml`. The important reproducibility settings are:

```yaml
training:
  seed: 42
  deterministic: true
  device: cuda
```

Run the complete Kedro pipeline:

```powershell
.venv\Scripts\kedro.exe run
```

The run produces training and validation losses, segmentation metrics, the best checkpoint, validation prediction overlays, and an MLflow experiment run.

### What is made deterministic?

The configured seed is applied to Python's random number generator, NumPy, PyTorch, CUDA, DataLoader workers, and the weighted training sampler. When `deterministic: true`, PyTorch also requests deterministic algorithm behavior where supported.

### What is recorded?

Each MLflow run records the dataset manifest, image size, model architecture, learning rate, loss weights, batch size, epoch count, seed, and selected device. It also stores the training history, best checkpoint, configuration file, and validation prediction examples.

### Reproducibility limitations

Deterministic settings improve repeatability but do not guarantee identical results across every machine. Small differences may still occur because of different PyTorch or CUDA versions, GPU hardware, CPU versus GPU execution, unsupported nondeterministic operations, or changes to the dataset and split manifest.

For the closest comparison, use the same Python version, dependency versions, dataset manifest, seed, image size, device type, and hyperparameters.

## ONNX export

The registered MLflow candidate can be exported to ONNX with:

```powershell
.venv\Scripts\python.exe scripts\export_onnx.py
```

The script loads `models:/factoryvision-segmentation@candidate`, exports the U-Net graph and trained weights, validates the ONNX file, and compares one ONNX Runtime prediction with the original PyTorch prediction. The exported model keeps the segmentation output as logits:

```text
Input:  images — (batch_size, 3, 256, 640) float32
Output: logits — (batch_size, 1, 256, 640) float32
```

The batch dimension is dynamic, while the image height and width remain the configured `256 x 640` size. The generated file is stored at `artifacts/models/factoryvision-segmentation.onnx` and remains outside Git because generated artifacts are ignored. The initial verification produced a maximum absolute difference of `2.29e-05` between PyTorch and ONNX Runtime outputs.

## PyTorch versus ONNX Runtime benchmark

Run the controlled CPU benchmark with:

```powershell
.venv\Scripts\python.exe scripts\benchmark_inference.py
```

The benchmark uses the registered PyTorch model and the exported ONNX model with the same `1 x 3 x 256 x 640` input, one CPU thread, 10 warm-up runs, and 30 measured runs by default. It reports mean latency, p95 latency, throughput, model size, and output agreement. The PyTorch size is the complete MLflow model bundle; the ONNX size is the generated ONNX file.

The initial benchmark used 5 warm-up runs and 20 measured runs on CPU:

| Runtime | Model size | Mean latency | p95 latency | Throughput |
| --- | ---: | ---: | ---: | ---: |
| PyTorch via MLflow | 40.311 MB | 830.17 ms | 842.97 ms | 1.205 images/s |
| ONNX Runtime | 29.613 MB | 669.14 ms | 683.99 ms | 1.494 images/s |

In this comparison, ONNX Runtime reduced mean latency by approximately 19% and increased throughput by approximately 24%. The maximum absolute output difference was `2.29e-05`, so the speed comparison did not come at the cost of a meaningful prediction change. These numbers are CPU- and hardware-specific; the benchmark should be rerun on the target deployment machine.

## FastAPI inference service

The local API loads the exported ONNX model once at startup and exposes three endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Reports whether the service, model, and prediction storage are ready. |
| `GET /model-info` | Returns the model alias, runtime, tensor shapes, and threshold. |
| `POST /predict` | Accepts an image and returns a defect mask, score, and bounding box. |

Start the service from the repository root:

```powershell
.venv\Scripts\uvicorn.exe factoryvision.api.main:app --host 127.0.0.1 --port 8000
```

Then open the interactive API documentation at `http://127.0.0.1:8000/docs`.

Example requests from PowerShell:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/model-info
curl.exe -X POST http://127.0.0.1:8000/predict -F "file=@path\\to\\inspection.png"
```

The prediction response contains `defect_probability`, `defect_area_fraction`, `has_defect`, a `bounding_box` when defect pixels are present, and a base64-encoded PNG in `mask_base64`. The mask is produced at `256 x 640`, while the original image dimensions are also returned. The service uses the same aspect-preserving resize, letterboxing, RGB conversion, and `[0, 1]` normalization as the training dataset loader.

The default model path is `artifacts/models/factoryvision-segmentation.onnx`. It can be overridden without changing code:

```powershell
$env:FACTORYVISION_ONNX_MODEL = "C:\\models\\factoryvision-segmentation.onnx"
$env:FACTORYVISION_THRESHOLD = "0.5"
```

## PostgreSQL prediction storage

The API persists one row for each inspected image in a `predictions` table. The row contains a SHA-256 image identifier, timestamp, model name and alias, defect score, defect area, bounding-box coordinates, inference latency, and a `success` or `error` status. The binary mask remains in the API response rather than being stored as a large Base64 value in PostgreSQL.

The default database URL is:

```text
postgresql+psycopg://factoryvision:factoryvision@localhost:5432/factoryvision
```

For local development, start PostgreSQL with Docker:

```powershell
docker run --name factoryvision-postgres `
  -e POSTGRES_DB=factoryvision `
  -e POSTGRES_USER=factoryvision `
  -e POSTGRES_PASSWORD=factoryvision `
  -p 5432:5432 `
  -d postgres:16
```

The API creates the table at startup. To point it at another database, set:

```powershell
$env:FACTORYVISION_DATABASE_URL = "postgresql+psycopg://user:password@host:5432/database"
```

After a successful `/predict` request, the API stores the prediction metadata. If inference fails after the image has been read, it stores an error record with the measured latency and error message. The `/health` response includes `storage_ready` so database readiness is visible.

## Run the API, PostgreSQL, and Airflow with Docker

The repository includes a multi-stage [Dockerfile](Dockerfile), an Airflow image definition at `docker/Dockerfile.airflow`, and a `docker-compose.yml`. Compose starts the API, PostgreSQL, and the Airflow scheduler, DAG processor, and API server. Airflow uses the same PostgreSQL service for its metadata and for FactoryVision prediction records; the Airflow tables and application tables coexist in that database. The serving and batch images install runtime dependencies only; the large training and CUDA-related dependencies remain available through the optional `training` extra for local development.

The ONNX model is intentionally not stored in Git because it is a generated artifact. Generate it locally first:

```powershell
.venv\Scripts\python.exe scripts\export_onnx.py
```

Then build and start the stack:

```powershell
docker-compose up --build
```

Docker Compose v2 users can use the equivalent `docker compose up --build` command.

The API is available at `http://127.0.0.1:8000`, the interactive documentation is at `http://127.0.0.1:8000/docs`, and Airflow is available at `http://127.0.0.1:8080`. The local Airflow username is `airflow`. The first initialization generates its password in `simple_auth_manager_passwords.json`; read it with:

```powershell
docker-compose exec airflow-api-server cat /opt/airflow/simple_auth_manager_passwords.json
```

This Simple Auth Manager setup is for local development only; use a production-grade auth manager before exposing Airflow beyond the local machine.

The model is mounted read-only from `artifacts/models/` into the API container. If that file is missing, the API container will fail its health check with a clear model-not-found error. Stop the stack while preserving database rows with:

```powershell
docker-compose down
```

To remove the PostgreSQL data volume as well, use `docker-compose down --volumes`.

### Trigger the batch DAG

The DAG is defined in [`airflow/dags/factoryvision_batch.py`](airflow/dags/factoryvision_batch.py) with the ID `factoryvision_batch_inference`. It runs daily at midnight UTC and is paused when first created so that a local user can inspect the configuration before processing files.

Place new inspection images in:

```text
data/batch/incoming/
```

Then open Airflow, unpause `factoryvision_batch_inference`, and trigger it manually from the UI. The DAG performs these tasks in order:

```text
discover_images
    -> validate_images
    -> run_batch_inference
    -> persist_results
    -> generate_summary
```

The discovery task uses the image's SHA-256 content ID and skips images already present in `predictions`. Validation confirms that OpenCV can decode each file. Inference writes a compact hand-off file, `persist_results` writes prediction metadata to PostgreSQL, and the final task writes a JSON summary. Each run produces files under:

```text
artifacts/batch/<airflow-run-id>/predictions.json
artifacts/batch/<airflow-run-id>/summary.json
```

The batch image directory, artifact directory, database URL, model path, and optional maximum number of images can be changed with `FACTORYVISION_BATCH_IMAGE_DIR`, `FACTORYVISION_BATCH_ARTIFACT_DIR`, `FACTORYVISION_DATABASE_URL`, `FACTORYVISION_ONNX_MODEL`, and `FACTORYVISION_BATCH_MAX_IMAGES`. The implementation can also be tested without Airflow because the reusable functions live in `src/factoryvision/batch/`.

The Compose setup follows the structure of the [official Airflow Docker guidance](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/), while using a project-specific image so the DAG can import FactoryVision's ONNX and PostgreSQL code.

## PySpark prediction analytics

The Day 4 analytics job is [`spark/daily_prediction_statistics.py`](spark/daily_prediction_statistics.py). It reads the PostgreSQL `predictions` table through Spark JDBC and produces daily statistics grouped by prediction date, model name, model alias, and derived defect status. Successful predictions are classified as `defect` or `no_defect` using the configured probability threshold; failed inference records are retained as `error`.

Install the optional PySpark dependency:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[analytics]"
```

Run the job locally with the PostgreSQL JDBC driver:

```powershell
$env:FACTORYVISION_DATABASE_URL = "postgresql+psycopg://factoryvision:factoryvision@localhost:5432/factoryvision"
.venv\Scripts\spark-submit.cmd `
  --packages org.postgresql:postgresql:42.7.4 `
  spark\daily_prediction_statistics.py
```

The default Parquet output is written to `artifacts/spark/daily_prediction_statistics`. The output includes prediction counts, average defect probability, average latency, total/successful predictions, defect/error counts, defect rate, and error rate. Use `--format csv` or `--format json` for an interchange format. The JDBC driver is provided with `--packages` because Spark's JDBC reader requires a database-specific driver on its classpath. PySpark 3.5.6 is used here because it supports the project's Python 3.11 environment and Java 8/11/17; Apache Spark's installation and JDBC documentation describe these compatibility and driver requirements.

On Windows, Spark 3.5.6 may also require Hadoop `winutils.exe` with `HADOOP_HOME` set when staging JDBC jars or writing output files. If that is not configured, run the job in a Linux/Docker Spark environment. The DataFrame aggregation tests do not require `winutils.exe`.

## Definition of done

- A fresh clone can reproduce training or inference from documented commands.
- Dataset and model versions are explicit.
- At least three experiments are visible in MLflow.
- Metrics and example predictions are published in the README.
- The API has automated tests.
- The Docker image builds in GitHub Actions.
- Batch inference is orchestrated by Airflow.
- Predictions are persisted in SQL.
- Grafana displays live service and model metrics.
- An ONNX benchmark is included.
- Cloud or Kubernetes deployment is demonstrated.
- The README includes the architecture and a short demo.

## Portfolio summary

FactoryVision demonstrates the full lifecycle of an applied computer-vision system: data engineering, segmentation, evaluation, experiment tracking, model versioning, optimized inference, APIs, batch orchestration, deployment, CI/CD, and observability.

The project is designed to complement research-heavy computer-vision experience with practical production ML and MLOps engineering.

# Load-test baseline

This is a small local baseline for the Docker Compose API. It is intended to
show how the measurement works, not to claim production capacity.

| Measurement | Result |
| --- | ---: |
| Endpoint | `POST /predict` |
| Input | `assets/dataset/non_defect_sample.jpg` |
| Requests | 20 |
| Concurrency | 4 |
| Successful requests | 20 |
| Failed requests | 0 |
| Error rate | 0.0% |
| Throughput | 4.8279 requests/s |
| Mean latency | 803.3074 ms |
| p50 latency | 669.3392 ms |
| p95 latency | 1368.4134 ms |

The test ran on 2026-08-31 against `http://127.0.0.1:8000/predict`, with the
API and PostgreSQL running through Docker Compose. Latency is measured from
the client before the multipart request is sent until the response body is
read, so it includes upload, API processing, database persistence, and the
response transfer.

The reusable command is:

```powershell
.venv\Scripts\python.exe scripts\load_test.py `
  --url http://127.0.0.1:8000/predict `
  --image assets\dataset\non_defect_sample.jpg `
  --requests 20 `
  --concurrency 4 `
  --output artifacts\load-test\report.json
```

The full machine-readable result is written to
`artifacts/load-test/report.json`, which is intentionally ignored because it
is generated output. To compare configurations, change `--requests` or
`--concurrency` and record the new environment and values alongside the
result.

# MLOps Batch Processing Pipeline

A production-style batch pipeline that ingests financial time-series data, computes a rolling-mean signal, and emits structured JSON metrics — with full observability, reproducibility, and Docker support.

---

## Project Overview

| Concern | Approach |
|---|---|
| **Reproducibility** | Seeded NumPy RNG via `config.yaml` |
| **Observability** | Structured timestamped logs (`run.log`) + JSON metrics (`metrics.json`) |
| **Deployment readiness** | Docker-first; no hard-coded paths; configurable via CLI flags |
| **Error handling** | Typed exceptions; `metrics.json` always written (success *and* failure) |
| **Config-driven** | All tuneable parameters live in `config.yaml` |

---

## Project Structure

```
mlops_task/
├── run.py           # Pipeline entry point
├── config.yaml      # Runtime configuration
├── data.csv         # Input dataset (OHLCV, 10 000 rows)
├── requirements.txt # Pinned Python dependencies
├── Dockerfile       # Production container definition
├── README.md        # This file
├── metrics.json     # Generated output — metrics summary
└── run.log          # Generated output — structured log
```

---

## Setup — Local Execution

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

All four flags are **required**; no paths are hard-coded.

---

## Docker Usage

### Build the image

```bash
docker build -t mlops-task .
```

### Run the container

```bash
docker run --rm mlops-task
```

The container bundles `data.csv` and `config.yaml`, runs the pipeline, prints `metrics.json` to stdout, and exits with code `0` on success or non-zero on failure.

#### Retrieve output files from a running container

```bash
# Start container, keep it alive, copy outputs, then remove
docker run --name mlops-run mlops-task
docker cp mlops-run:/app/metrics.json .
docker cp mlops-run:/app/run.log .
docker rm mlops-run
```

Or use a bind-mount to persist outputs directly to your host:

```bash
docker run --rm \
  -v "$(pwd)/output:/app/output" \
  mlops-task python run.py \
    --input    data.csv \
    --config   config.yaml \
    --output   output/metrics.json \
    --log-file output/run.log
```

---

## Configuration Reference

`config.yaml`:

```yaml
seed: 42      # Integer — NumPy random seed for reproducibility
window: 5     # Integer ≥ 1 — rolling-mean window size
version: "v1" # String — pipeline version tag written to metrics
```

All fields are required. Missing or mis-typed fields cause a validation error with a descriptive message in `metrics.json`.

---

## Processing Logic

1. Load `close` column from input CSV.
2. Compute `rolling_mean` with the configured `window`.
3. Generate binary `signal`:  
   `signal = 1 if close > rolling_mean else 0`  
   (first `window - 1` rows where rolling mean is undefined receive `signal = 0`)
4. Compute `signal_rate = mean(signal)` over all rows.

---

## Sample `metrics.json` — Success

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.499,
  "latency_ms": 312,
  "seed": 42,
  "status": "success"
}
```

## Sample `metrics.json` — Failure

```json
{
  "version": "unknown",
  "status": "error",
  "error_message": "Input file not found: missing.csv"
}
```

---

## Dependencies

| Package  | Version | Purpose |
|----------|---------|---------|
| pandas   | 2.0.3   | CSV I/O and DataFrame operations |
| numpy    | 1.24.4  | Vectorised signal computation and RNG seeding |
| pyyaml   | 6.0.1   | YAML config parsing |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0`  | Pipeline completed successfully |
| `1`  | Pipeline failed (see `metrics.json` and `run.log` for details) |

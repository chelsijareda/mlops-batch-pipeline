# MLOps Batch Processing Pipeline

## Overview

This project demonstrates a production-style MLOps batch processing pipeline using Python.

Features:
- reproducible execution
- structured logging
- dataset validation
- deterministic outputs
- rolling mean signal generation
- Docker support
- structured exception handling
- metrics reporting

## Local Execution

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

## Docker Build

```bash
docker build -t mlops-task .
```

## Docker Run

```bash
docker run --rm mlops-task
```

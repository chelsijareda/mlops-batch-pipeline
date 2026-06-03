# ---------------------------------------------------------------
# MLOps Batch Processing Pipeline
# Base image: python:3.9-slim (production-style, minimal footprint)
# ---------------------------------------------------------------
FROM python:3.9-slim

# Prevent interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install Python dependencies first (layer-cached separately from source)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and required data files
COPY run.py      .
COPY config.yaml .
COPY data.csv    .

# Default command: run the pipeline with fixed in-container paths.
# All outputs land in /app so they are accessible via docker cp or volume mounts.
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]
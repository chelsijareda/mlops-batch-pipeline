import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class ConfigValidationError(Exception):
    pass


class DatasetValidationError(Exception):
    pass


def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_path: str) -> dict:
    config_file = Path(config_path)

    if not config_file.exists():
        raise ConfigValidationError(f"Config file not found: {config_path}")

    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Invalid YAML format: {e}")

    required_fields = ["seed", "window", "version"]

    for field in required_fields:
        if field not in config:
            raise ConfigValidationError(f"Missing required config field: {field}")

    return config


def validate_and_load_dataset(input_path: str) -> pd.DataFrame:
    input_file = Path(input_path)

    if not input_file.exists():
        raise DatasetValidationError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        raise DatasetValidationError(f"Unreadable CSV file: {e}")

    if df.empty:
        raise DatasetValidationError("CSV file is empty")

    if "close" not in df.columns:
        raise DatasetValidationError("Missing required column: close")

    return df


def process_data(df: pd.DataFrame, window: int) -> pd.DataFrame:
    logging.info("Starting rolling mean processing")

    df["rolling_mean"] = df["close"].rolling(window=window).mean()

    logging.info("Generating trading signals")

    df["signal"] = np.where(
        df["close"] > df["rolling_mean"],
        1,
        0
    )

    df["signal"] = df["signal"].fillna(0)

    return df


def write_metrics(output_path: str, metrics: dict):
    with open(output_path, "w") as file:
        json.dump(metrics, file, indent=4)


def main():
    parser = argparse.ArgumentParser(description="MLOps Batch Processing Pipeline")

    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-file", required=True)

    args = parser.parse_args()

    setup_logging(args.log_file)

    logging.info("Batch job started")

    metrics = {}

    try:
        start_time = time.time()

        logging.info("Loading configuration")

        config = load_config(args.config)

        logging.info("Configuration validation successful")

        seed = config["seed"]
        window = config["window"]
        version = config["version"]

        np.random.seed(seed)

        logging.info(f"Reproducible seed set to {seed}")

        logging.info("Loading dataset")

        df = validate_and_load_dataset(args.input)

        logging.info(f"Rows loaded: {len(df)}")

        processed_df = process_data(df, window)

        signal_rate = float(processed_df["signal"].mean())

        latency_ms = int((time.time() - start_time) * 1000)

        metrics = {
            "version": version,
            "rows_processed": int(len(processed_df)),
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success"
        }

        logging.info(f"Metrics summary: {metrics}")

        write_metrics(args.output, metrics)

        logging.info("Batch job completed successfully")

        print(json.dumps(metrics, indent=4))

        sys.exit(0)

    except Exception as e:
        logging.exception("Pipeline execution failed")

        version = "unknown"

        try:
            if "config" in locals():
                version = config.get("version", "unknown")
        except Exception:
            pass

        metrics = {
            "version": version,
            "status": "error",
            "error_message": str(e)
        }

        write_metrics(args.output, metrics)

        print(json.dumps(metrics, indent=4))

        sys.exit(1)


if __name__ == "__main__":
    main()

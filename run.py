"""
MLOps Batch Processing Pipeline
Entry point for the signal generation pipeline.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    """Configure root logger to write to both file and stdout."""
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_FIELDS = {"seed", "window", "version"}


def load_config(config_path: str) -> dict:
    """Load and validate YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file is empty or not a valid YAML mapping.")

    missing = REQUIRED_CONFIG_FIELDS - config.keys()
    if missing:
        raise ValueError(f"Config is missing required fields: {sorted(missing)}")

    if not isinstance(config["seed"], int):
        raise ValueError(f"Config field 'seed' must be an integer, got: {type(config['seed']).__name__}")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError(f"Config field 'window' must be a positive integer, got: {config['window']}")
    if not isinstance(config["version"], str) or not config["version"].strip():
        raise ValueError("Config field 'version' must be a non-empty string.")

    return config


# ---------------------------------------------------------------------------
# Dataset loading & validation
# ---------------------------------------------------------------------------

def load_dataset(input_path: str) -> pd.DataFrame:
    """Load and validate the input CSV dataset."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file '{input_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Input CSV '{input_path}' is empty (no rows).")

    if "close" not in df.columns:
        raise ValueError(
            f"Input CSV is missing required column 'close'. "
            f"Found columns: {list(df.columns)}"
        )

    return df


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling mean and binary signal on the 'close' column.

    Signal = 1 if close > rolling_mean else 0.
    Rows where rolling_mean is NaN (first window-1 rows) receive signal = 0.
    """
    close = df["close"].astype(float)
    rolling_mean = close.rolling(window=window, min_periods=window).mean()

    signal = np.where(
        rolling_mean.notna() & (close > rolling_mean),
        1,
        0,
    )

    result = df.copy()
    result["rolling_mean"] = rolling_mean
    result["signal"] = signal
    return result


# ---------------------------------------------------------------------------
# Metrics I/O
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict) -> None:
    """Serialize metrics dict to JSON file."""
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def build_success_metrics(version: str, rows: int, signal_rate: float,
                           latency_ms: int, seed: int) -> dict:
    return {
        "version": version,
        "rows_processed": rows,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": latency_ms,
        "seed": seed,
        "status": "success",
    }


def build_error_metrics(version: str, error_message: str) -> dict:
    return {
        "version": version,
        "status": "error",
        "error_message": error_message,
    }


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str, config_path: str,
                 output_path: str, log_file: str) -> int:
    """
    Execute the full batch pipeline.
    Returns exit code: 0 on success, 1 on failure.
    """
    logger = setup_logging(log_file)
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("MLOps Batch Pipeline — job started")
    logger.info(f"  input   : {input_path}")
    logger.info(f"  config  : {config_path}")
    logger.info(f"  output  : {output_path}")
    logger.info(f"  log     : {log_file}")
    logger.info("=" * 60)

    version = "unknown"

    try:
        # --- Config ---
        logger.info("Loading and validating configuration ...")
        config = load_config(config_path)
        version = config["version"]
        seed = config["seed"]
        window = config["window"]
        logger.info(
            f"Config validated — version={version}, seed={seed}, window={window}"
        )

        # --- Reproducibility ---
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")

        # --- Dataset ---
        logger.info(f"Loading dataset from '{input_path}' ...")
        df = load_dataset(input_path)
        logger.info(f"Rows loaded: {len(df):,}  |  Columns: {list(df.columns)}")

        # --- Processing ---
        logger.info(f"Computing rolling mean (window={window}) ...")
        result_df = compute_signals(df, window)
        logger.info("Rolling mean computation complete.")

        logger.info("Generating binary signals ...")
        rows_processed = len(result_df)
        signal_rate = float(result_df["signal"].mean())
        logger.info(
            f"Signal generation complete — "
            f"rows={rows_processed:,}, signal_rate={signal_rate:.4f}"
        )

        # --- Metrics ---
        elapsed_ms = int((time.time() - start_time) * 1000)
        metrics = build_success_metrics(
            version=version,
            rows=rows_processed,
            signal_rate=signal_rate,
            latency_ms=elapsed_ms,
            seed=seed,
        )
        write_metrics(output_path, metrics)
        logger.info(f"Metrics written to '{output_path}'")
        logger.info(
            f"Summary — rows={rows_processed:,}, "
            f"signal_rate={signal_rate:.4f}, latency_ms={elapsed_ms}"
        )

        logger.info("Job completed successfully.")
        logger.info("=" * 60)
        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"Pipeline failed: {error_msg}", exc_info=True)

        try:
            error_metrics = build_error_metrics(version=version, error_message=error_msg)
            write_metrics(output_path, error_metrics)
            logger.info(f"Error metrics written to '{output_path}'")
            print(json.dumps(error_metrics, indent=2))
        except Exception as write_exc:
            logger.error(f"Failed to write error metrics: {write_exc}")

        logger.info("Job ended with FAILURE.")
        logger.info("=" * 60)
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLOps Batch Processing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",    required=True, help="Path to input CSV file")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path for log file output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = run_pipeline(
        input_path=args.input,
        config_path=args.config,
        output_path=args.output,
        log_file=args.log_file,
    )
    sys.exit(exit_code)
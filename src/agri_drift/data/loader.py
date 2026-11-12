"""Data ingestion and format loaders for Agri-Drift."""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import json

from agri_drift.data.schema import validate_dataframe_schema


class DataLoader:
    """Ingests multi-sensor time-series datasets from CSV or Parquet files with metadata tracking."""

    @staticmethod
    def load_csv(
        file_path: str | Path,
        timestamp_col: str = "timestamp",
        timestamp_format: Optional[str] = None,
        metadata_file: Optional[str | Path] = None,
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Load time-series from CSV, parse timestamps, and validate schema."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_csv(path)
        if timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{timestamp_col}' not found in CSV columns: {df.columns.tolist()}")

        df["timestamp"] = pd.to_datetime(df[timestamp_col], format=timestamp_format, utc=True)
        if timestamp_col != "timestamp":
            df = df.drop(columns=[timestamp_col])

        df = df.sort_values("timestamp").reset_index(drop=True)
        validate_dataframe_schema(df)

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "rows": len(df),
            "start_time": str(df["timestamp"].min()),
            "end_time": str(df["timestamp"].max()),
        }

        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, "r") as f:
                metadata["manifest"] = json.load(f)

        return df, metadata

    @staticmethod
    def load_parquet(file_path: str | Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Load time-series from Parquet format."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        validate_dataframe_schema(df)

        metadata = {
            "source_path": str(path),
            "rows": len(df),
            "start_time": str(df["timestamp"].min()),
            "end_time": str(df["timestamp"].max()),
        }
        return df, metadata

    @staticmethod
    def save_parquet(df: pd.DataFrame, file_path: str | Path) -> None:
        """Save normalized time-series to Parquet."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

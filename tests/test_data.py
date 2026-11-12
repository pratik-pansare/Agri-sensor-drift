"""Tests for data schema, synthetic generator, loader, and preprocessor."""

import pytest
import pandas as pd
import numpy as np

from agri_drift.data.schema import validate_dataframe_schema, HealthState, FaultType
from agri_drift.data.synthetic_generator import SyntheticHydrologyGenerator
from agri_drift.data.preprocessor import Preprocessor
from agri_drift.data.loader import DataLoader


def test_synthetic_hydrology_generation():
    generator = SyntheticHydrologyGenerator(days=10, interval_minutes=15, random_seed=42)
    df, events = generator.generate()

    assert len(df) == 10 * 96  # 96 steps per day
    assert "timestamp" in df.columns
    assert "sensor_1" in df.columns
    assert "sensor_4" in df.columns
    assert validate_dataframe_schema(df) is True

    # Sensors should be physically within plausible range
    for i in range(1, 5):
        assert df[f"sensor_{i}"].min() >= 2.0
        assert df[f"sensor_{i}"].max() <= 55.0

    # There should be at least one rain or irrigation event generated
    assert len(events) > 0


def test_chronological_split_no_leakage():
    generator = SyntheticHydrologyGenerator(days=10, interval_minutes=15, random_seed=42)
    df, _ = generator.generate()

    train_df, val_df, test_df = Preprocessor.chronological_split(df, 0.70, 0.15, 0.15)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    # Check strict chronological order
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()


def test_csv_save_and_load(tmp_path):
    generator = SyntheticHydrologyGenerator(days=3, interval_minutes=60, random_seed=42)
    df, _ = generator.generate()

    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    loaded_df, meta = DataLoader.load_csv(csv_path)
    assert len(loaded_df) == len(df)
    assert meta["rows"] == len(df)
    assert "sensor_1" in loaded_df.columns

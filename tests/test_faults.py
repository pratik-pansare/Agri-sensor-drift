"""Tests for fault injection engine and manifest management."""

import pytest
import numpy as np
import pandas as pd

from agri_drift.data.synthetic_generator import SyntheticHydrologyGenerator
from agri_drift.faults.injector import FaultInjector
from agri_drift.faults.manifest import FaultManifest
from agri_drift.data.schema import FaultType


@pytest.fixture
def baseline_df():
    gen = SyntheticHydrologyGenerator(days=10, interval_minutes=15, random_seed=42)
    df, _ = gen.generate()
    return df


def test_inject_drift(baseline_df):
    injector = FaultInjector(random_seed=42, interval_minutes=15)
    # Inject 2.0 % VWC / day drift for 2 days (192 steps)
    corrupted_df, record = injector.inject_drift(
        baseline_df,
        sensor_id="sensor_2",
        start_idx=100,
        duration_steps=192,
        rate_vwc_per_day=2.0,
    )

    assert record.fault_type == FaultType.DRIFT.value
    assert record.target_sensor == "sensor_2"
    # End value should be greater by approx 4.0 % VWC (2 days * 2.0%/day)
    clean_diff = baseline_df.loc[291, "sensor_2"]
    corrupted_diff = corrupted_df.loc[291, "sensor_2"]
    expected_drift = 2.0 * (191 / 96.0)
    assert np.isclose(corrupted_diff - clean_diff, expected_drift, atol=1e-3)


def test_inject_bias(baseline_df):
    injector = FaultInjector(random_seed=42)
    corrupted_df, record = injector.inject_bias(
        baseline_df,
        sensor_id="sensor_1",
        start_idx=50,
        duration_steps=100,
        offset_vwc=5.5,
    )

    assert record.fault_type == FaultType.BIAS.value
    delta = corrupted_df.loc[50:149, "sensor_1"] - baseline_df.loc[50:149, "sensor_1"]
    assert np.allclose(delta, 5.5)


def test_inject_stuck_at(baseline_df):
    injector = FaultInjector(random_seed=42)
    corrupted_df, record = injector.inject_stuck_at(
        baseline_df,
        sensor_id="sensor_3",
        start_idx=80,
        duration_steps=50,
        mode="freeze_current",
    )

    assert record.fault_type == FaultType.STUCK_AT.value
    stuck_vals = corrupted_df.loc[80:129, "sensor_3"].values
    assert len(np.unique(stuck_vals)) == 1
    assert stuck_vals[0] == baseline_df.loc[80, "sensor_3"]


def test_inject_spikes(baseline_df):
    injector = FaultInjector(random_seed=42)
    corrupted_df, record = injector.inject_spikes(
        baseline_df,
        sensor_id="sensor_4",
        start_idx=50,
        duration_steps=100,
        amplitude_vwc=20.0,
        n_spikes=3,
    )

    assert record.fault_type == FaultType.SPIKE.value
    diff = np.abs(corrupted_df.loc[50:149, "sensor_4"] - baseline_df.loc[50:149, "sensor_4"])
    assert (diff > 5.0).sum() >= 1


def test_inject_noise(baseline_df):
    injector = FaultInjector(random_seed=42)
    corrupted_df, record = injector.inject_noise(
        baseline_df,
        sensor_id="sensor_2",
        start_idx=100,
        duration_steps=200,
        std_multiplier=5.0,
    )

    assert record.fault_type == FaultType.NOISE.value
    clean_std = np.std(np.diff(baseline_df.loc[100:299, "sensor_2"]))
    corrupted_std = np.std(np.diff(corrupted_df.loc[100:299, "sensor_2"]))
    assert corrupted_std > clean_std * 2.0


def test_manifest_ground_truth_mask(baseline_df):
    injector = FaultInjector(random_seed=42)
    _, rec = injector.inject_bias(baseline_df, "sensor_1", 10, 20, 5.0)

    manifest = FaultManifest([rec])
    mask = manifest.get_ground_truth_mask(baseline_df, "sensor_1")
    assert mask.iloc[10:30].all()
    assert not mask.iloc[0:10].any()
    assert not mask.iloc[30:].any()

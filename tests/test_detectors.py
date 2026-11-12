"""Tests for threshold, neighbour regression, CUSUM, and Page-Hinkley detectors."""

import pytest
import numpy as np
import pandas as pd

from agri_drift.data.synthetic_generator import SyntheticHydrologyGenerator
from agri_drift.detectors.threshold import ThresholdDetector
from agri_drift.detectors.neighbour import NeighbourResidualDetector
from agri_drift.detectors.cusum import CusumDetector
from agri_drift.detectors.page_hinkley import PageHinkleyDetector


@pytest.fixture
def synthetic_data():
    gen = SyntheticHydrologyGenerator(days=15, interval_minutes=15, random_seed=42)
    df, _ = gen.generate()
    train_df = df.iloc[:1000]
    test_df = df.iloc[1000:].reset_index(drop=True)
    return train_df, test_df


def test_threshold_detector():
    detector = ThresholdDetector(min_vwc=3.0, max_vwc=50.0, max_rate=5.0)

    # Normal reading
    res = detector.update("sensor_1", 25.0)
    assert not res.is_anomaly

    # Normal next reading
    res = detector.update("sensor_1", 26.0)
    assert not res.is_anomaly

    # Rate violation: sudden jump by 10.0
    res = detector.update("sensor_1", 36.0)
    assert res.is_anomaly
    assert "rate of change" in res.reason.lower()

    # Out of range reading
    res = detector.update("sensor_2", 1.0)
    assert res.is_anomaly
    assert "range" in res.reason.lower()


def test_cusum_detector():
    cusum = CusumDetector(allowance_k=0.5, threshold_h=5.0, reset_on_alert=True)

    # Healthy standard normal noise: should not trigger alarm
    rng = np.random.default_rng(42)
    healthy_steps = rng.normal(0, 1.0, 100)
    for z in healthy_steps:
        res = cusum.update(float(z))
        assert not res.is_change

    # Positive drift signal (shift to mean +2.0)
    drift_steps = rng.normal(2.0, 1.0, 30)
    detected = False
    for z in drift_steps:
        res = cusum.update(float(z))
        if res.is_change:
            detected = True
            assert res.direction == "positive"
            break
    assert detected, "CUSUM failed to detect positive drift"


def test_page_hinkley_detector():
    ph = PageHinkleyDetector(delta=0.5, threshold_lambda=15.0)

    rng = np.random.default_rng(42)
    # Healthy stationary noise
    for _ in range(80):
        res = ph.update(float(rng.normal(0, 1.0)))
        assert not res.is_change

    # Persistent shift
    detected = False
    for _ in range(50):
        res = ph.update(float(rng.normal(2.5, 1.0)))
        if res.is_change:
            detected = True
            break
    assert detected, "Page-Hinkley failed to detect persistent mean shift"


def test_neighbour_residual_and_contamination_defense(synthetic_data):
    train_df, test_df = synthetic_data
    sensors = ["sensor_1", "sensor_2", "sensor_3", "sensor_4"]

    neighbour_det = NeighbourResidualDetector(target_sensors=sensors)
    neighbour_det.fit(train_df)

    # Check that residual on healthy test data is low
    row = {s: float(test_df.loc[0, s]) for s in sensors}
    pred, raw_res, std_res = neighbour_det.predict("sensor_1", row)
    assert abs(raw_res) < 2.0  # Raw error within 2% VWC
    assert abs(std_res) < 3.0  # Within 3 standard deviations

    # Contamination defense test:
    # If sensor_2 is corrupted by +15% VWC
    corrupted_row = row.copy()
    corrupted_row["sensor_2"] += 15.0

    # Without excluding sensor_2, sensor_1's prediction is pulled
    pred_unmasked, raw_unmasked, _ = neighbour_det.predict("sensor_1", corrupted_row)

    # With excluding sensor_2, sensor_1 uses leave-one-out submodel
    pred_masked, raw_masked, _ = neighbour_det.predict(
        "sensor_1", corrupted_row, excluded_sensors={"sensor_2"}
    )

    # The masked prediction should be much closer to true healthy prediction
    assert abs(pred_masked - pred) < abs(pred_unmasked - pred)


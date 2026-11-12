"""Tests for physics consistency detector and hybrid decision fusion engine."""

from datetime import datetime, timedelta
import pytest
import numpy as np
import pandas as pd

from agri_drift.data.schema import HealthState, FaultType, WaterEvent
from agri_drift.data.synthetic_generator import SyntheticHydrologyGenerator
from agri_drift.detectors.threshold import ThresholdDetector
from agri_drift.detectors.neighbour import NeighbourResidualDetector
from agri_drift.detectors.physics import PhysicsConsistencyDetector
from agri_drift.detectors.hybrid import HybridDecisionEngine


@pytest.fixture
def test_setup():
    gen = SyntheticHydrologyGenerator(days=15, interval_minutes=15, random_seed=42)
    df, _ = gen.generate()
    train_df = df.iloc[:1000]

    sensors = ["sensor_1", "sensor_2", "sensor_3", "sensor_4"]
    thresh_det = ThresholdDetector()
    neighbour_det = NeighbourResidualDetector(target_sensors=sensors)
    neighbour_det.fit(train_df)

    physics_det = PhysicsConsistencyDetector(target_sensors=sensors)

    engine = HybridDecisionEngine(
        target_sensors=sensors,
        threshold_detector=thresh_det,
        neighbour_detector=neighbour_det,
        physics_detector=physics_det,
        confirmation_window_steps=2,
    )
    return engine, df


def test_hybrid_healthy_state(test_setup):
    engine, df = test_setup
    now = pd.to_datetime(df.loc[1005, "timestamp"])

    # Clean healthy readings from held-out healthy data
    readings = {s: float(df.loc[1005, s]) for s in engine.target_sensors}
    decisions = engine.process_step(now, readings)

    for s, dec in decisions.items():
        assert dec.state == HealthState.HEALTHY
        assert dec.suspected_subtype == FaultType.NONE
        assert "normally" in dec.explanation


def test_hybrid_drift_confirmation(test_setup):
    engine, df = test_setup
    t = datetime(2026, 5, 1, 12, 0)

    # Gradually drift sensor_2 upwards while others stay flat
    readings = {"sensor_1": 25.0, "sensor_2": 25.0, "sensor_3": 25.0, "sensor_4": 25.0}

    confirmed = False
    for step in range(30):
        t += timedelta(minutes=15)
        # Drift sensor_2 by +0.3% per step
        readings["sensor_2"] += 0.3
        decisions = engine.process_step(t, readings)

        if decisions["sensor_2"].state == HealthState.CONFIRMED_FAULT:
            confirmed = True
            assert decisions["sensor_2"].suspected_subtype in [FaultType.DRIFT, FaultType.BIAS]
            assert "Sensor sensor_2" in decisions["sensor_2"].explanation
            break

    assert confirmed, "Hybrid engine failed to confirm gradual drift"


def test_physics_consistency_response(test_setup):
    engine, df = test_setup
    physics_det = engine.physics_det
    t0 = datetime(2026, 5, 1, 8, 0)

    # Pre-event readings
    readings_pre = {"sensor_1": 20.0, "sensor_2": 20.0, "sensor_3": 20.0, "sensor_4": 20.0}
    event = WaterEvent(timestamp=t0, event_type="irrigation", amount_mm=12.0)
    physics_det.register_event(event, readings_pre)

    # During response window, sensors 1, 2, 3 respond (+6% VWC), but sensor_4 does not respond (stuck at 20.0)
    t_after = t0 + timedelta(hours=4, minutes=15)
    readings_post = {"sensor_1": 26.0, "sensor_2": 26.2, "sensor_3": 25.8, "sensor_4": 20.1}

    scores = physics_det.update(t_after, readings_post)
    # Sensor 4 should have accumulated suspicion
    assert scores["sensor_4"] > 0.30
    assert scores["sensor_1"] == 0.0

"""Tests for hardware abstraction, mock sensors, database storage, and replay streaming."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from agri_drift.hardware.mock_reader import MockSensorReader
from agri_drift.hardware.ads1115_reader import ADS1115SensorReader
from agri_drift.storage.database import DatabaseManager
from agri_drift.data.schema import HealthDecision, HealthState, FaultType
from agri_drift.streaming.replay import ReplayStreamer
from agri_drift.data.synthetic_generator import SyntheticHydrologyGenerator
from agri_drift.detectors.threshold import ThresholdDetector
from agri_drift.detectors.neighbour import NeighbourResidualDetector
from agri_drift.detectors.hybrid import HybridDecisionEngine


def test_mock_sensor_reader():
    reader = MockSensorReader()
    assert reader.is_connected() is True

    readings = reader.read_channels()
    assert len(readings) == 4
    for s, (voltage, vwc) in readings.items():
        assert 0.0 <= voltage <= 3.3
        assert 10.0 <= vwc <= 45.0

    # Test disconnection simulation
    reader.simulate_disconnect("sensor_2")
    readings = reader.read_channels()
    assert np.isnan(readings["sensor_2"][0])
    assert np.isnan(readings["sensor_2"][1])
    assert not np.isnan(readings["sensor_1"][0])

    # Reconnect
    reader.simulate_reconnect("sensor_2")
    readings = reader.read_channels()
    assert not np.isnan(readings["sensor_2"][0])


def test_ads1115_reader_laptop_safety():
    # ADS1115 reader should initialize cleanly without throwing on laptop
    reader = ADS1115SensorReader()
    assert reader.is_connected() is False

    with pytest.raises(RuntimeError, match="Physical ADS1115 hardware is not connected"):
        reader.read_channels()


def test_database_and_replay(tmp_path):
    db_file = tmp_path / "test_agri.db"
    db = DatabaseManager(db_path=db_file)

    now = datetime(2026, 5, 1, 10, 0)
    db.record_reading(now, "sensor_1", 24.5, raw_voltage=2.1)

    readings_df = db.get_recent_readings()
    assert len(readings_df) == 1
    assert readings_df.iloc[0]["sensor_id"] == "sensor_1"
    assert readings_df.iloc[0]["vwc_percent"] == 24.5

    # Test alert storage
    dec = HealthDecision(
        timestamp=now,
        sensor_id="sensor_3",
        state=HealthState.CONFIRMED_FAULT,
        suspected_subtype=FaultType.DRIFT,
        explanation="Sensor 3 calibration drift confirmed",
    )
    db.record_decision(dec)

    alerts_df = db.get_recent_alerts()
    assert len(alerts_df) == 1
    assert alerts_df.iloc[0]["sensor_id"] == "sensor_3"
    assert alerts_df.iloc[0]["state"] == "CONFIRMED_FAULT"


def test_replay_streamer_execution(tmp_path):
    gen = SyntheticHydrologyGenerator(days=2, interval_minutes=60, random_seed=42)
    df, _ = gen.generate()

    sensors = ["sensor_1", "sensor_2", "sensor_3", "sensor_4"]
    thresh = ThresholdDetector()
    neighbour = NeighbourResidualDetector(target_sensors=sensors)
    neighbour.fit(df.iloc[:24])

    engine = HybridDecisionEngine(
        target_sensors=sensors,
        threshold_detector=thresh,
        neighbour_detector=neighbour,
    )

    db = DatabaseManager(db_path=tmp_path / "replay.db")
    streamer = ReplayStreamer(df=df.iloc[24:34], engine=engine, db_manager=db)

    # Step 5 times
    for _ in range(5):
        decs = streamer.step()
        assert decs is not None

    assert streamer.current_index == 5
    readings = db.get_recent_readings()
    assert len(readings) == 5 * 4  # 5 steps * 4 sensors

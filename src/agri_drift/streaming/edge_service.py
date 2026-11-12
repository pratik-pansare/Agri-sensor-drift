"""Edge logging daemon service for Raspberry Pi deployment."""

import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
import yaml

from agri_drift.hardware.ads1115_reader import ADS1115SensorReader
from agri_drift.hardware.mock_reader import MockSensorReader
from agri_drift.detectors.threshold import ThresholdDetector
from agri_drift.detectors.neighbour import NeighbourResidualDetector
from agri_drift.detectors.hybrid import HybridDecisionEngine
from agri_drift.storage.database import DatabaseManager
from agri_drift.alerts.manager import AlertManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agri_drift.edge_service")


def run_service(config_path: str = "configs/default.yaml") -> None:
    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    db_path = cfg["storage"].get("database_path", "data/agri_drift.db")
    db = DatabaseManager(db_path=db_path)
    alert_mgr = AlertManager(db_manager=db, enable_console=True)

    target_sensors = cfg["sensors"]["target_sensors"]

    # Select hardware or mock reader
    reader = ADS1115SensorReader()
    if not reader.is_connected():
        logger.warning("Physical ADS1115 not connected. Defaulting to MockSensorReader for demonstration.")
        reader = MockSensorReader()

    # Initialize detectors
    thresh_det = ThresholdDetector(
        min_vwc=cfg["detectors"]["threshold"]["min_vwc"],
        max_vwc=cfg["detectors"]["threshold"]["max_vwc"],
        max_rate=cfg["detectors"]["threshold"]["max_rate"],
    )

    neighbour_det = NeighbourResidualDetector(target_sensors=target_sensors)
    # If historical data exists in DB, fit models; otherwise use identity prior
    recent_df = db.get_recent_readings(limit_per_sensor=200)
    if not recent_df.empty and len(recent_df["sensor_id"].unique()) == len(target_sensors):
        pivot = recent_df.pivot(index="timestamp", columns="sensor_id", values="vwc_percent").dropna()
        if len(pivot) > 20:
            neighbour_det.fit(pivot)
            logger.info("Neighbour regression models successfully calibrated on recent database history.")

    engine = HybridDecisionEngine(
        target_sensors=target_sensors,
        threshold_detector=thresh_det,
        neighbour_detector=neighbour_det,
        confirmation_window_steps=cfg["hybrid"].get("confirmation_window_steps", 3),
        recovery_window_steps=cfg["hybrid"].get("recovery_window_steps", 5),
    )

    interval_sec = cfg.get("poll_interval_seconds", 15)  # Can be fast for test, or 900 for 15-min field
    logger.info(f"Edge monitoring service active. Polling every {interval_sec} seconds...")

    try:
        while True:
            now = datetime.now(timezone.utc)
            raw_readings = reader.read_channels()

            vwc_dict = {}
            for s, (voltage, vwc) in raw_readings.items():
                db.record_reading(now, s, vwc=vwc, raw_voltage=voltage)
                vwc_dict[s] = vwc

            decisions = engine.process_step(now, vwc_dict)
            for s, dec in decisions.items():
                alert_mgr.dispatch(dec)

            time.sleep(interval_sec)
    except KeyboardInterrupt:
        logger.info("Edge monitoring service stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()
    run_service(args.config)

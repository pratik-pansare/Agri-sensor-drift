"""Chronological streaming replay engine for agricultural IoT time-series."""

import time
from typing import Dict, Any, List, Optional
import pandas as pd

from agri_drift.data.schema import HealthDecision
from agri_drift.detectors.hybrid import HybridDecisionEngine
from agri_drift.storage.database import DatabaseManager
from agri_drift.alerts.manager import AlertManager


class ReplayStreamer:
    """Streams time-series readings tick-by-tick through the hybrid detection engine into SQLite."""

    def __init__(
        self,
        df: pd.DataFrame,
        engine: HybridDecisionEngine,
        db_manager: Optional[DatabaseManager] = None,
        alert_manager: Optional[AlertManager] = None,
        speed_factor: float = 0.0,  # 0.0 for max speed / batch, >0 for delayed replay
    ):
        self.df = df.sort_values("timestamp").reset_index(drop=True)
        self.engine = engine
        self.db_manager = db_manager
        self.alert_manager = alert_manager
        self.speed_factor = speed_factor

        self.current_index: int = 0
        self.total_steps: int = len(self.df)
        self.is_paused: bool = False

    def step(self) -> Optional[Dict[str, HealthDecision]]:
        """Advance exactly one time step and return diagnostic decisions."""
        if self.current_index >= self.total_steps:
            return None

        row = self.df.iloc[self.current_index]
        ts = pd.to_datetime(row["timestamp"])

        readings = {s: float(row[s]) for s in self.engine.target_sensors if s in row}

        # Store readings in database
        if self.db_manager:
            for s, val in readings.items():
                self.db_manager.record_reading(timestamp=ts, sensor_id=s, vwc=val)

        # Process through hybrid decision engine
        decisions = self.engine.process_step(ts, readings)

        # Store decisions and dispatch alerts
        if self.db_manager:
            for s, dec in decisions.items():
                self.db_manager.record_decision(dec)

        if self.alert_manager:
            for s, dec in decisions.items():
                self.alert_manager.dispatch(dec)

        self.current_index += 1

        if self.speed_factor > 0.0:
            delay = 1.0 / self.speed_factor
            time.sleep(delay)

        return decisions

    def run_all(self) -> List[Dict[str, HealthDecision]]:
        """Process all remaining steps synchronously."""
        all_decisions = []
        while self.current_index < self.total_steps:
            decs = self.step()
            if decs:
                all_decisions.append(decs)
        return all_decisions

    def reset(self) -> None:
        """Reset replay position to the beginning."""
        self.current_index = 0


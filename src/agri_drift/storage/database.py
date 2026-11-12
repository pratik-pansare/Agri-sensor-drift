"""SQLite database storage layer for readings, detector states, and alert logs."""

import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from agri_drift.data.schema import HealthState, FaultType, HealthDecision


class DatabaseManager:
    """Manages SQLite storage for agricultural IoT time-series and diagnostic logs."""

    def __init__(self, db_path: str | Path = "data/agri_drift.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Create necessary relational tables if not present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    sensor_id TEXT NOT NULL,
                    vwc_percent REAL NOT NULL,
                    raw_voltage REAL,
                    quality_flag TEXT DEFAULT 'G'
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detector_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    sensor_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    subtype TEXT NOT NULL,
                    residual REAL,
                    cusum_pos REAL,
                    cusum_neg REAL,
                    page_hinkley REAL,
                    physics_suspicion REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    sensor_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    subtype TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    confirmed INTEGER DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings (timestamp, sensor_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_states_ts ON detector_states (timestamp, sensor_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts (timestamp);
            """)
            conn.commit()

    def record_reading(
        self,
        timestamp: datetime,
        sensor_id: str,
        vwc: float,
        raw_voltage: Optional[float] = None,
        quality_flag: str = "G",
    ) -> None:
        """Insert a single sensor observation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO readings (timestamp, sensor_id, vwc_percent, raw_voltage, quality_flag)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(timestamp), sensor_id, float(vwc), raw_voltage, quality_flag),
            )
            conn.commit()

    def record_decision(self, decision: HealthDecision) -> None:
        """Insert detector state and conditionally create alert entry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO detector_states (
                    timestamp, sensor_id, state, subtype, residual,
                    cusum_pos, cusum_neg, page_hinkley, physics_suspicion
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(decision.timestamp),
                    decision.sensor_id,
                    decision.state.value,
                    decision.suspected_subtype.value,
                    decision.residual,
                    decision.cusum_pos,
                    decision.cusum_neg,
                    decision.page_hinkley_stat,
                    decision.physics_suspicion,
                ),
            )

            # Record alert if suspected or confirmed fault
            if decision.state in [HealthState.SUSPECTED_FAULT, HealthState.CONFIRMED_FAULT]:
                severity = "HIGH" if decision.state == HealthState.CONFIRMED_FAULT else "MEDIUM"
                is_confirmed = 1 if decision.state == HealthState.CONFIRMED_FAULT else 0
                cursor.execute(
                    """
                    INSERT INTO alerts (timestamp, sensor_id, state, subtype, severity, explanation, confirmed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(decision.timestamp),
                        decision.sensor_id,
                        decision.state.value,
                        decision.suspected_subtype.value,
                        severity,
                        decision.explanation,
                        is_confirmed,
                    ),
                )
            conn.commit()

    def get_recent_readings(self, limit_per_sensor: int = 100) -> pd.DataFrame:
        """Fetch latest sensor readings as DataFrame."""
        with self._get_connection() as conn:
            query = f"""
                SELECT timestamp, sensor_id, vwc_percent, raw_voltage
                FROM readings
                ORDER BY timestamp DESC
                LIMIT {limit_per_sensor * 4}
            """
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
            return df

    def get_recent_alerts(self, limit: int = 50) -> pd.DataFrame:
        """Fetch latest diagnostic alerts."""
        with self._get_connection() as conn:
            query = f"""
                SELECT timestamp, sensor_id, state, subtype, severity, explanation, confirmed
                FROM alerts
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)
            return df

    def get_detector_history(self, sensor_id: str, limit: int = 200) -> pd.DataFrame:
        """Fetch time-series of residuals and CUSUM stats for a specific sensor."""
        with self._get_connection() as conn:
            query = f"""
                SELECT timestamp, state, subtype, residual, cusum_pos, cusum_neg, page_hinkley, physics_suspicion
                FROM detector_states
                WHERE sensor_id = '{sensor_id}'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
            return df


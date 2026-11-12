"""Unified data schema and core domain entities for Agri-Drift."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import pandas as pd


class HealthState(str, Enum):
    """Primary sensor health states."""
    HEALTHY = "HEALTHY"
    SUSPECTED_FAULT = "SUSPECTED_FAULT"
    CONFIRMED_FAULT = "CONFIRMED_FAULT"
    UNCERTAIN = "UNCERTAIN"
    OFFLINE_OR_MISSING = "OFFLINE_OR_MISSING"


class FaultType(str, Enum):
    """Classes of sensor faults."""
    NONE = "none"
    DRIFT = "drift"
    BIAS = "bias"
    STUCK_AT = "stuck_at"
    SPIKE = "spike"
    NOISE = "noise"


@dataclass(frozen=True)
class WaterEvent:
    """Record of an environmental or management water addition event."""
    timestamp: datetime
    event_type: str  # "rain" or "irrigation"
    amount_mm: float
    affected_sensors: List[str] = field(default_factory=lambda: ["all"])
    notes: Optional[str] = None


@dataclass
class SensorReading:
    """Individual sensor reading record."""
    timestamp: datetime
    sensor_id: str
    vwc_percent: float
    raw_voltage: Optional[float] = None
    is_valid: bool = True
    quality_flag: str = "G"  # "G" for Good, "M" for Missing, "A" for Anomaly


@dataclass
class HealthDecision:
    """Diagnostic health assessment output by the hybrid decision engine."""
    timestamp: datetime
    sensor_id: str
    state: HealthState
    suspected_subtype: FaultType = FaultType.NONE
    confidence: float = 1.0
    explanation: str = "Sensor operating normally"
    residual: Optional[float] = None
    cusum_pos: Optional[float] = None
    cusum_neg: Optional[float] = None
    page_hinkley_stat: Optional[float] = None
    physics_suspicion: Optional[float] = None


REQUIRED_DATA_COLUMNS = [
    "timestamp",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "rain_mm",
    "irrigation_mm",
    "ambient_temp",
    "relative_humidity",
]


def validate_dataframe_schema(df: pd.DataFrame) -> bool:
    """Verify that a DataFrame conforms to the canonical Agri-Drift multi-sensor schema."""
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame missing required 'timestamp' column.")
    
    # Check for at least 3 sensor columns
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    if len(sensor_cols) < 3:
        raise ValueError(f"At least 3 sensor columns required, found {len(sensor_cols)}: {sensor_cols}")
    
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError(f"Timestamp column must be datetime, found {df['timestamp'].dtype}")
    
    return True

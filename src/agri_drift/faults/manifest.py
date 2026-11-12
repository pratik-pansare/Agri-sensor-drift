"""Fault manifest serialization and ground-truth metadata management."""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import json
import pandas as pd

from agri_drift.data.schema import FaultType


@dataclass
class FaultRecord:
    """Ground truth record of an injected fault episode."""
    fault_id: str
    target_sensor: str
    fault_type: str  # e.g. "drift", "bias", "stuck_at", "spike", "noise"
    start_time: str  # ISO-8601 UTC
    end_time: str    # ISO-8601 UTC
    duration_hours: float
    parameters: Dict[str, Any]
    random_seed: int
    clean_mean: float
    faulty_mean: float


class FaultManifest:
    """Manages ground-truth labels and episode records for experimental evaluation."""

    def __init__(self, records: List[FaultRecord] = None):
        self.records: List[FaultRecord] = records or []

    def add_record(self, record: FaultRecord) -> None:
        self.records.append(record)

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in self.records]
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "FaultManifest":
        p = Path(path)
        with open(p, "r") as f:
            data = json.load(f)
        records = [FaultRecord(**item) for item in data]
        return cls(records=records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert manifest records into a pandas DataFrame."""
        return pd.DataFrame([asdict(r) for r in self.records])

    def get_ground_truth_mask(
        self,
        df: pd.DataFrame,
        sensor_id: str,
        fault_type: str = None,
    ) -> pd.Series:
        """Generate a boolean Series indicating where the specified sensor was in a fault state."""
        timestamps = pd.to_datetime(df["timestamp"], utc=True)
        mask = pd.Series(False, index=df.index)

        for r in self.records:
            if r.target_sensor != sensor_id:
                continue
            if fault_type and r.fault_type != fault_type:
                continue

            r_start = pd.to_datetime(r.start_time, utc=True)
            r_end = pd.to_datetime(r.end_time, utc=True)
            mask |= (timestamps >= r_start) & (timestamps <= r_end)

        return mask


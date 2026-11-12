"""Fixed range and rate-of-change threshold baseline detector."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class ThresholdAlert:
    is_anomaly: bool
    reason: Optional[str] = None
    value: float = 0.0
    delta: Optional[float] = None


class ThresholdDetector:
    """Baseline detector using static physical range and maximum rate-of-change rules."""

    def __init__(
        self,
        min_vwc: float = 3.0,
        max_vwc: float = 55.0,
        max_rate: float = 8.0,
        persistence_steps: int = 1,
    ):
        self.min_vwc = min_vwc
        self.max_vwc = max_vwc
        self.max_rate = max_rate
        self.persistence_steps = persistence_steps

        # Stateful tracking per sensor: last_value, consecutive_violations
        self._last_values: Dict[str, float] = {}
        self._consecutive_counts: Dict[str, int] = {}

    def reset(self, sensor_id: Optional[str] = None) -> None:
        """Reset internal state for a sensor or all sensors."""
        if sensor_id:
            self._last_values.pop(sensor_id, None)
            self._consecutive_counts.pop(sensor_id, None)
        else:
            self._last_values.clear()
            self._consecutive_counts.clear()

    def update(self, sensor_id: str, value: float) -> ThresholdAlert:
        """Process a single streaming reading for a given sensor."""
        if np.isnan(value):
            return ThresholdAlert(is_anomaly=True, reason="MISSING_OR_NAN", value=value)

        # Range check
        range_violation = value < self.min_vwc or value > self.max_vwc

        # Rate of change check
        rate_violation = False
        delta = None
        if sensor_id in self._last_values:
            delta = abs(value - self._last_values[sensor_id])
            if delta > self.max_rate:
                rate_violation = True

        self._last_values[sensor_id] = value

        if range_violation or rate_violation:
            self._consecutive_counts[sensor_id] = self._consecutive_counts.get(sensor_id, 0) + 1
        else:
            self._consecutive_counts[sensor_id] = 0

        is_flagged = self._consecutive_counts[sensor_id] >= self.persistence_steps

        reason = None
        if is_flagged:
            if range_violation and rate_violation:
                reason = f"Out of range ({value:.1f}%) and excessive rate of change ({delta:.1f}%)"
            elif range_violation:
                reason = f"Out of valid range [{self.min_vwc}, {self.max_vwc}]%: {value:.1f}%"
            elif rate_violation:
                reason = f"Excessive rate of change: {delta:.1f}% > {self.max_rate}%"

        return ThresholdAlert(
            is_anomaly=is_flagged,
            reason=reason,
            value=value,
            delta=delta,
        )

    def detect_batch(self, df: pd.DataFrame, sensor_id: str) -> pd.DataFrame:
        """Run batch threshold detection across a historical DataFrame."""
        self.reset(sensor_id)
        results = []
        for val in df[sensor_id].values:
            alert = self.update(sensor_id, float(val))
            results.append({
                "threshold_flag": alert.is_anomaly,
                "threshold_reason": alert.reason,
                "threshold_delta": alert.delta,
            })
        return pd.DataFrame(results)

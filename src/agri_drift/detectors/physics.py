"""Hydrological physics consistency detector based on rainfall and irrigation events."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from agri_drift.data.schema import WaterEvent


class PhysicsConsistencyDetector:
    """Evaluates whether sensors exhibit physically expected moisture surges following recorded rain/irrigation.

    If water is added to the soil and neighbouring sensors record a rise, a non-responsive sensor
    accumulates a suspicion score rather than triggering an immediate binary fault.
    """

    def __init__(
        self,
        target_sensors: List[str],
        response_window_hours: float = 4.0,
        min_expected_response_vwc: float = 1.5,
        suspicion_increment: float = 0.35,
        suspicion_decay: float = 0.10,
        suspicion_threshold: float = 0.70,
    ):
        self.target_sensors = target_sensors
        self.response_window_hours = response_window_hours
        self.min_expected_response_vwc = min_expected_response_vwc
        self.suspicion_increment = suspicion_increment
        self.suspicion_decay = suspicion_decay
        self.suspicion_threshold = suspicion_threshold

        # Active events tracking: list of WaterEvent
        self.events: List[WaterEvent] = []
        # Pre-event baseline readings: event_id -> sensor_id -> vwc
        self.pre_event_levels: Dict[int, Dict[str, float]] = {}
        # Max observed post-event readings: event_id -> sensor_id -> max_vwc
        self.max_observed_post: Dict[int, Dict[str, float]] = {}
        # Running suspicion score per sensor
        self.suspicion_scores: Dict[str, float] = {s: 0.0 for s in target_sensors}

    def register_event(self, event: WaterEvent, current_readings: Dict[str, float]) -> None:
        """Register a new rainfall or irrigation event with baseline readings."""
        event_idx = len(self.events)
        self.events.append(event)
        self.pre_event_levels[event_idx] = {
            s: current_readings.get(s, np.nan) for s in self.target_sensors
        }
        self.max_observed_post[event_idx] = {
            s: current_readings.get(s, np.nan) for s in self.target_sensors
        }

    def update(
        self,
        current_time: datetime,
        current_readings: Dict[str, float],
    ) -> Dict[str, float]:
        """Update physics consistency scores based on observed response to active events.

        Returns a dictionary of current suspicion scores for each sensor.
        """
        if not self.events:
            # Transparent skip when no physical events have been recorded
            return self.suspicion_scores.copy()

        # Update post-event maximums and evaluate closed windows
        active_indices = []
        for idx, event in enumerate(self.events):
            window_end = event.timestamp + timedelta(hours=self.response_window_hours)

            # Update maximum observed reading in window
            for s in self.target_sensors:
                val = current_readings.get(s, np.nan)
                if not np.isnan(val):
                    self.max_observed_post[idx][s] = max(
                        self.max_observed_post[idx].get(s, val), val
                    )

            if current_time >= window_end:
                # Window closed: evaluate response delta
                self._evaluate_window(idx)
            else:
                active_indices.append(idx)

        # Decay suspicion slightly over quiet periods
        for s in self.target_sensors:
            self.suspicion_scores[s] = max(0.0, self.suspicion_scores[s] - 0.005)

        return self.suspicion_scores.copy()

    def _evaluate_window(self, event_idx: int) -> None:
        """Evaluate pre-post delta across sensors after response window ends."""
        deltas: Dict[str, float] = {}
        for s in self.target_sensors:
            pre = self.pre_event_levels.get(event_idx, {}).get(s, np.nan)
            post_max = self.max_observed_post.get(event_idx, {}).get(s, np.nan)
            if not np.isnan(pre) and not np.isnan(post_max):
                deltas[s] = post_max - pre
            else:
                deltas[s] = 0.0

        # Determine if neighbours responded
        valid_deltas = [d for s, d in deltas.items() if not np.isnan(d)]
        median_delta = float(np.median(valid_deltas)) if valid_deltas else 0.0

        for s in self.target_sensors:
            d = deltas.get(s, 0.0)
            if median_delta >= self.min_expected_response_vwc:
                # Soil definitely received water; check individual sensor
                if d < 0.4 * self.min_expected_response_vwc:
                    # Non-responsive probe: increase suspicion
                    self.suspicion_scores[s] = min(1.0, self.suspicion_scores[s] + self.suspicion_increment)
                else:
                    # Responsive: decrease suspicion
                    self.suspicion_scores[s] = max(0.0, self.suspicion_scores[s] - self.suspicion_decay)


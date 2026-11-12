"""Hybrid decision engine fusing spatial residuals, sequential statistics, and hydrological physics."""

from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from agri_drift.data.schema import HealthState, FaultType, HealthDecision
from agri_drift.detectors.threshold import ThresholdDetector, ThresholdAlert
from agri_drift.detectors.neighbour import NeighbourResidualDetector
from agri_drift.detectors.cusum import CusumDetector
from agri_drift.detectors.page_hinkley import PageHinkleyDetector
from agri_drift.detectors.physics import PhysicsConsistencyDetector


class HybridDecisionEngine:
    """Interpretable multi-evidence decision engine for agricultural sensor health diagnostics.

    Fuses:
    1. Fixed range and rate-of-change boundaries
    2. Spatial neighbour residual deviations
    3. CUSUM & Page-Hinkley sequential change detection
    4. Rain and irrigation physical response consistency
    5. Persistence and recovery filters with contamination isolation
    """

    def __init__(
        self,
        target_sensors: List[str],
        threshold_detector: ThresholdDetector,
        neighbour_detector: NeighbourResidualDetector,
        confirmation_window_steps: int = 3,
        recovery_window_steps: int = 5,
        physics_detector: Optional[PhysicsConsistencyDetector] = None,
        cusum_allowance_k: float = 0.5,
        cusum_threshold_h: float = 5.0,
        ph_threshold_lambda: float = 15.0,
    ):
        self.target_sensors = target_sensors
        self.threshold_det = threshold_detector
        self.neighbour_det = neighbour_detector
        self.physics_det = physics_detector
        self.confirmation_window = confirmation_window_steps
        self.recovery_window = recovery_window_steps

        # Instantiate sequential detectors per sensor
        self.cusum_dets: Dict[str, CusumDetector] = {
            s: CusumDetector(allowance_k=cusum_allowance_k, threshold_h=cusum_threshold_h)
            for s in target_sensors
        }
        self.ph_dets: Dict[str, PageHinkleyDetector] = {
            s: PageHinkleyDetector(threshold_lambda=ph_threshold_lambda)
            for s in target_sensors
        }

        # Operational state per sensor
        self.current_states: Dict[str, HealthState] = {s: HealthState.HEALTHY for s in target_sensors}
        self.consecutive_suspected: Dict[str, int] = {s: 0 for s in target_sensors}
        self.consecutive_healthy: Dict[str, int] = {s: 0 for s in target_sensors}
        self.recent_deltas: Dict[str, deque] = {s: deque(maxlen=6) for s in target_sensors}
        self.recent_residuals: Dict[str, deque] = {s: deque(maxlen=6) for s in target_sensors}
        self.active_faults: Set[str] = set()

    def process_step(
        self,
        timestamp: datetime,
        readings: Dict[str, float],
    ) -> Dict[str, HealthDecision]:
        """Process a single synchronous time step across all target sensors."""
        decisions: Dict[str, HealthDecision] = {}

        # Update physics detector if configured
        physics_scores = {}
        if self.physics_det:
            physics_scores = self.physics_det.update(timestamp, readings)

        for s in self.target_sensors:
            val = readings.get(s, np.nan)

            # Check 1: Missing or NaN
            if np.isnan(val):
                self.current_states[s] = HealthState.OFFLINE_OR_MISSING
                self.active_faults.add(s)
                decisions[s] = HealthDecision(
                    timestamp=timestamp,
                    sensor_id=s,
                    state=HealthState.OFFLINE_OR_MISSING,
                    suspected_subtype=FaultType.NONE,
                    explanation=f"Sensor {s} reading is missing or disconnected",
                )
                continue

            # Check 2: Range and rate thresholds
            thresh_alert: ThresholdAlert = self.threshold_det.update(s, val)
            if thresh_alert.delta is not None:
                self.recent_deltas[s].append(thresh_alert.delta)

            # Check 3: Spatial neighbour residual (excluding currently active faulty sensors)
            _, raw_res, std_res = self.neighbour_det.predict(
                target_sensor=s,
                readings=readings,
                excluded_sensors=self.active_faults - {s},
            )
            if not np.isnan(std_res):
                self.recent_residuals[s].append(std_res)

            # Check 4: Sequential CUSUM and Page-Hinkley
            cusum_res = self.cusum_dets[s].update(std_res if not np.isnan(std_res) else 0.0)
            ph_res = self.ph_dets[s].update(std_res if not np.isnan(std_res) else 0.0)

            # Check 5: Physics suspicion
            phys_susp = physics_scores.get(s, 0.0)

            # Evidence integration
            has_thresh_anomaly = thresh_alert.is_anomaly
            has_drift_anomaly = cusum_res.is_change or ph_res.is_change
            has_residual_anomaly = not np.isnan(std_res) and abs(std_res) > 3.0
            has_physics_anomaly = phys_susp >= 0.70

            is_anomalous = has_thresh_anomaly or has_drift_anomaly or has_residual_anomaly or has_physics_anomaly

            # State transition logic
            if is_anomalous:
                self.consecutive_suspected[s] += 1
                self.consecutive_healthy[s] = 0

                if self.consecutive_suspected[s] >= self.confirmation_window:
                    new_state = HealthState.CONFIRMED_FAULT
                    self.active_faults.add(s)
                else:
                    new_state = HealthState.SUSPECTED_FAULT
            else:
                self.consecutive_healthy[s] += 1
                if self.consecutive_healthy[s] >= self.recovery_window:
                    new_state = HealthState.HEALTHY
                    self.consecutive_suspected[s] = 0
                    self.active_faults.discard(s)
                else:
                    new_state = self.current_states[s]

            self.current_states[s] = new_state

            # Subtype classification
            subtype = self._classify_subtype(
                s=s,
                val=val,
                thresh_alert=thresh_alert,
                std_res=std_res,
                cusum_res=cusum_res,
                ph_res=ph_res,
                phys_susp=phys_susp,
            )

            # Generate plain-English explanation
            explanation = self._generate_explanation(
                sensor_id=s,
                state=new_state,
                subtype=subtype,
                thresh_alert=thresh_alert,
                std_res=std_res,
                cusum_res=cusum_res,
                phys_susp=phys_susp,
            )

            decisions[s] = HealthDecision(
                timestamp=timestamp,
                sensor_id=s,
                state=new_state,
                suspected_subtype=subtype,
                explanation=explanation,
                residual=raw_res,
                cusum_pos=cusum_res.s_pos,
                cusum_neg=cusum_res.s_neg,
                page_hinkley_stat=ph_res.ph_stat,
                physics_suspicion=phys_susp,
            )

        return decisions

    def _classify_subtype(
        self,
        s: str,
        val: float,
        thresh_alert: ThresholdAlert,
        std_res: float,
        cusum_res,
        ph_res,
        phys_susp: float,
    ) -> FaultType:
        """Classify suspected fault subtype based on characteristic signal dynamics."""
        if self.current_states[s] == HealthState.HEALTHY:
            return FaultType.NONE

        deltas = list(self.recent_deltas[s])
        res_history = list(self.recent_residuals[s])

        # Stuck-at: consecutive deltas near 0 or fixed at saturation bounds
        if val <= 1.0 or val >= 54.0 or (len(deltas) >= 4 and max(deltas[-4:]) < 1e-4):
            return FaultType.STUCK_AT

        # Spike: large threshold rate or large single residual return
        if thresh_alert.delta is not None and thresh_alert.delta > 10.0:
            return FaultType.SPIKE

        # Noise: high residual variance without strong monotonic CUSUM
        if len(res_history) >= 4 and np.std(res_history[-4:]) > 2.5 and not cusum_res.is_change:
            return FaultType.NOISE

        # Drift: CUSUM alarm or Page-Hinkley alarm with sustained residual accumulation
        if cusum_res.is_change or ph_res.is_change:
            return FaultType.DRIFT

        # Bias: sustained offset
        if abs(std_res) > 2.5:
            return FaultType.BIAS

        return FaultType.NONE

    def _generate_explanation(
        self,
        sensor_id: str,
        state: HealthState,
        subtype: FaultType,
        thresh_alert: ThresholdAlert,
        std_res: float,
        cusum_res,
        phys_susp: float,
    ) -> str:
        """Generate human-readable, farmer-friendly explanation for alerts."""
        if state == HealthState.HEALTHY:
            return f"Sensor {sensor_id}: Operating normally. Readings consistent with neighbours."

        if subtype == FaultType.DRIFT:
            direction_str = f" ({cusum_res.direction} drift)" if cusum_res.direction else ""
            return (
                f"Sensor {sensor_id}: Possible calibration drift detected{direction_str}. "
                f"Measurements have diverged from healthy reference sensors (residual: {std_res:+.1f} σ). "
                f"Inspection or recalibration is recommended."
            )
        elif subtype == FaultType.STUCK_AT:
            return (
                f"Sensor {sensor_id}: Stuck-at fault detected. "
                f"Sensor reading is frozen or saturated without natural soil dynamics."
            )
        elif subtype == FaultType.BIAS:
            return (
                f"Sensor {sensor_id}: Constant bias offset detected. "
                f"Sudden shift of {std_res:+.1f} σ relative to neighbouring probes."
            )
        elif subtype == FaultType.SPIKE:
            return (
                f"Sensor {sensor_id}: Transient spike detected. "
                f"Anomalous instantaneous rate jump ({thresh_alert.delta:.1f}% VWC)."
            )
        elif subtype == FaultType.NOISE:
            return (
                f"Sensor {sensor_id}: Excessive measurement noise detected. "
                f"Signal variance has exceeded normal operating levels."
            )

        if phys_susp >= 0.70:
            return (
                f"Sensor {sensor_id}: Physics inconsistency detected. "
                f"Sensor failed to respond to recorded rain/irrigation while neighbours showed expected moisture rise."
            )

        return f"Sensor {sensor_id}: Suspected fault detected ({thresh_alert.reason or 'residual anomaly'})."

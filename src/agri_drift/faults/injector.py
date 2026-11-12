"""Deterministic fault-injection engine for agricultural IoT sensor networks."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from agri_drift.data.schema import FaultType
from agri_drift.faults.manifest import FaultManifest, FaultRecord


class FaultInjector:
    """Injects 5 canonical sensor fault types with exact ground truth manifests.

    Fault types:
    1. Linear drift (in % VWC per day)
    2. Constant bias offset (in % VWC)
    3. Stuck-at values (freeze current, ground zero, high saturation)
    4. Transient spike impulses
    5. Excessive / increasing noise
    """

    def __init__(self, random_seed: int = 42, interval_minutes: int = 15):
        self.random_seed = random_seed
        self.interval_minutes = interval_minutes
        self.steps_per_day = int(1440 / interval_minutes)

    def inject_drift(
        self,
        df: pd.DataFrame,
        sensor_id: str,
        start_idx: int,
        duration_steps: int,
        rate_vwc_per_day: float,
        fault_id: str = "fault_drift",
    ) -> Tuple[pd.DataFrame, FaultRecord]:
        """Inject linear drift: slope in % VWC per day."""
        df_mod = df.reset_index(drop=True).copy()
        end_idx = min(len(df_mod), start_idx + duration_steps)
        n_steps = end_idx - start_idx
        if n_steps <= 0:
            raise ValueError("Invalid injection window: start_idx >= len(df) or duration <= 0")

        slope_per_step = rate_vwc_per_day / self.steps_per_day
        drift_ramp = np.arange(n_steps) * slope_per_step

        clean_slice = df_mod.iloc[start_idx:end_idx, df_mod.columns.get_loc(sensor_id)].values
        corrupted_slice = np.clip(clean_slice + drift_ramp, 0.0, 100.0)
        df_mod.iloc[start_idx:end_idx, df_mod.columns.get_loc(sensor_id)] = corrupted_slice

        record = FaultRecord(
            fault_id=fault_id,
            target_sensor=sensor_id,
            fault_type=FaultType.DRIFT.value,
            start_time=str(df_mod.iloc[start_idx]["timestamp"]),
            end_time=str(df_mod.iloc[end_idx - 1]["timestamp"]),
            duration_hours=(n_steps * self.interval_minutes) / 60.0,
            parameters={"rate_vwc_per_day": rate_vwc_per_day, "slope_per_step": slope_per_step},
            random_seed=self.random_seed,
            clean_mean=float(np.mean(clean_slice)),
            faulty_mean=float(np.mean(corrupted_slice)),
        )
        return df_mod, record

    def inject_bias(
        self,
        df: pd.DataFrame,
        sensor_id: str,
        start_idx: int,
        duration_steps: int,
        offset_vwc: float,
        fault_id: str = "fault_bias",
    ) -> Tuple[pd.DataFrame, FaultRecord]:
        """Inject constant additive bias offset (% VWC)."""
        df_mod = df.reset_index(drop=True).copy()
        end_idx = min(len(df_mod), start_idx + duration_steps)
        n_steps = end_idx - start_idx

        clean_slice = df_mod.iloc[start_idx:end_idx, df_mod.columns.get_loc(sensor_id)].values
        corrupted_slice = np.clip(clean_slice + offset_vwc, 0.0, 100.0)
        df_mod.iloc[start_idx:end_idx, df_mod.columns.get_loc(sensor_id)] = corrupted_slice

        record = FaultRecord(
            fault_id=fault_id,
            target_sensor=sensor_id,
            fault_type=FaultType.BIAS.value,
            start_time=str(df_mod.iloc[start_idx]["timestamp"]),
            end_time=str(df_mod.iloc[end_idx - 1]["timestamp"]),
            duration_hours=(n_steps * self.interval_minutes) / 60.0,
            parameters={"offset_vwc": offset_vwc},
            random_seed=self.random_seed,
            clean_mean=float(np.mean(clean_slice)),
            faulty_mean=float(np.mean(corrupted_slice)),
        )
        return df_mod, record

    def inject_stuck_at(
        self,
        df: pd.DataFrame,
        sensor_id: str,
        start_idx: int,
        duration_steps: int,
        mode: str = "freeze_current",
        fixed_value: Optional[float] = None,
        fault_id: str = "fault_stuck",
    ) -> Tuple[pd.DataFrame, FaultRecord]:
        """Inject stuck-at reading."""
        df_mod = df.reset_index(drop=True).copy()
        end_idx = min(len(df_mod), start_idx + duration_steps)
        n_steps = end_idx - start_idx

        clean_slice = df_mod.iloc[start_idx:end_idx, df_mod.columns.get_loc(sensor_id)].values

        if mode == "freeze_current":
            stuck_val = clean_slice[0]
        elif mode == "ground_zero":
            stuck_val = 0.0
        elif mode == "saturation_high":
            stuck_val = 55.0
        elif mode == "custom":
            if fixed_value is None:
                raise ValueError("fixed_value must be provided for custom stuck_at mode")
            stuck_val = fixed_value
        else:
            raise ValueError(f"Unknown stuck_at mode: {mode}")

        corrupted_slice = np.full(n_steps, stuck_val)
        df_mod.iloc[start_idx:end_idx, df_mod.columns.get_loc(sensor_id)] = corrupted_slice

        record = FaultRecord(
            fault_id=fault_id,
            target_sensor=sensor_id,
            fault_type=FaultType.STUCK_AT.value,
            start_time=str(df_mod.iloc[start_idx]["timestamp"]),
            end_time=str(df_mod.iloc[end_idx - 1]["timestamp"]),
            duration_hours=(n_steps * self.interval_minutes) / 60.0,
            parameters={"mode": mode, "stuck_value": stuck_val},
            random_seed=self.random_seed,
            clean_mean=float(np.mean(clean_slice)),
            faulty_mean=float(stuck_val),
        )
        return df_mod, record

    def inject_spikes(
        self,
        df: pd.DataFrame,
        sensor_id: str,
        start_idx: int,
        duration_steps: int,
        amplitude_vwc: float = 18.0,
        n_spikes: int = 4,
        fault_id: str = "fault_spike",
    ) -> Tuple[pd.DataFrame, FaultRecord]:
        """Inject transient anomalous spikes."""
        rng = np.random.default_rng(self.random_seed)
        df_mod = df.reset_index(drop=True).copy()
        end_idx = min(len(df_mod), start_idx + duration_steps)
        n_steps = end_idx - start_idx

        clean_slice = df_mod.iloc[start_idx:end_idx, df_mod.columns.get_loc(sensor_id)].values.copy()
        corrupted_slice = clean_slice.copy()

        spike_indices = rng.choice(n_steps, size=min(n_spikes, n_steps), replace=False)
        for idx in spike_indices:
            sign = rng.choice([1.0, -1.0])
            corrupted_slice[idx] = np.clip(corrupted_slice[idx] + sign * amplitude_vwc, 0.0, 100.0)

        df_mod.iloc[start_idx:end_idx, df_mod.columns.get_loc(sensor_id)] = corrupted_slice

        record = FaultRecord(
            fault_id=fault_id,
            target_sensor=sensor_id,
            fault_type=FaultType.SPIKE.value,
            start_time=str(df_mod.iloc[start_idx]["timestamp"]),
            end_time=str(df_mod.iloc[end_idx - 1]["timestamp"]),
            duration_hours=(n_steps * self.interval_minutes) / 60.0,
            parameters={"amplitude_vwc": amplitude_vwc, "n_spikes": len(spike_indices)},
            random_seed=self.random_seed,
            clean_mean=float(np.mean(clean_slice)),
            faulty_mean=float(np.mean(corrupted_slice)),
        )
        return df_mod, record

    def inject_noise(
        self,
        df: pd.DataFrame,
        sensor_id: str,
        start_idx: int,
        duration_steps: int,
        std_multiplier: float = 4.0,
        base_std: float = 0.25,
        fault_id: str = "fault_noise",
    ) -> Tuple[pd.DataFrame, FaultRecord]:
        """Inject excessive/increasing measurement noise."""
        rng = np.random.default_rng(self.random_seed)
        df_mod = df.reset_index(drop=True).copy()
        end_idx = min(len(df_mod), start_idx + duration_steps)
        n_steps = end_idx - start_idx

        clean_slice = df_mod.iloc[start_idx:end_idx, df_mod.columns.get_loc(sensor_id)].values
        added_noise_std = base_std * (std_multiplier - 1.0)
        noise = rng.normal(0, added_noise_std, n_steps)
        corrupted_slice = np.clip(clean_slice + noise, 0.0, 100.0)
        df_mod.iloc[start_idx:end_idx, df_mod.columns.get_loc(sensor_id)] = corrupted_slice

        record = FaultRecord(
            fault_id=fault_id,
            target_sensor=sensor_id,
            fault_type=FaultType.NOISE.value,
            start_time=str(df_mod.iloc[start_idx]["timestamp"]),
            end_time=str(df_mod.iloc[end_idx - 1]["timestamp"]),
            duration_hours=(n_steps * self.interval_minutes) / 60.0,
            parameters={"std_multiplier": std_multiplier, "added_std": added_noise_std},
            random_seed=self.random_seed,
            clean_mean=float(np.mean(clean_slice)),
            faulty_mean=float(np.mean(corrupted_slice)),
        )
        return df_mod, record

    def inject_benchmark_suite(
        self,
        df: pd.DataFrame,
        episode_configs: List[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, FaultManifest]:
        """Inject a sequence of configured fault episodes into a clean dataset."""
        df_corrupted = df.copy()
        manifest = FaultManifest()

        for config in episode_configs:
            fault_type = config["fault_type"]
            sensor_id = config["sensor_id"]
            start_step = config.get("start_step")
            if start_step is None and "start_hour" in config:
                start_step = int(config["start_hour"] * 60 / self.interval_minutes)

            duration_steps = config.get("duration_steps")
            if duration_steps is None and "duration_hours" in config:
                duration_steps = int(config["duration_hours"] * 60 / self.interval_minutes)

            f_id = config.get("id", f"fault_{fault_type}_{sensor_id}")

            if fault_type == "drift":
                df_corrupted, rec = self.inject_drift(
                    df_corrupted,
                    sensor_id=sensor_id,
                    start_idx=start_step,
                    duration_steps=duration_steps,
                    rate_vwc_per_day=config["rate_vwc_per_day"],
                    fault_id=f_id,
                )
            elif fault_type == "bias":
                df_corrupted, rec = self.inject_bias(
                    df_corrupted,
                    sensor_id=sensor_id,
                    start_idx=start_step,
                    duration_steps=duration_steps,
                    offset_vwc=config["offset_vwc"],
                    fault_id=f_id,
                )
            elif fault_type == "stuck_at":
                df_corrupted, rec = self.inject_stuck_at(
                    df_corrupted,
                    sensor_id=sensor_id,
                    start_idx=start_step,
                    duration_steps=duration_steps,
                    mode=config.get("mode", "freeze_current"),
                    fixed_value=config.get("fixed_value"),
                    fault_id=f_id,
                )
            elif fault_type == "spike":
                df_corrupted, rec = self.inject_spikes(
                    df_corrupted,
                    sensor_id=sensor_id,
                    start_idx=start_step,
                    duration_steps=duration_steps,
                    amplitude_vwc=config.get("amplitude_vwc", 18.0),
                    fault_id=f_id,
                )
            elif fault_type == "noise":
                df_corrupted, rec = self.inject_noise(
                    df_corrupted,
                    sensor_id=sensor_id,
                    start_idx=start_step,
                    duration_steps=duration_steps,
                    std_multiplier=config.get("std_multiplier", 4.0),
                    fault_id=f_id,
                )
            else:
                raise ValueError(f"Unsupported fault type: {fault_type}")

            manifest.add_record(rec)

        return df_corrupted, manifest

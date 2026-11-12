"""Deterministic synthetic multi-sensor soil hydrology simulator."""

from datetime import datetime, timedelta
from typing import List, Tuple
import numpy as np
import pandas as pd

from agri_drift.data.schema import WaterEvent, validate_dataframe_schema


class SyntheticHydrologyGenerator:
    """Generates physically grounded agricultural soil moisture time series for 4 co-located sensor probes.

    Simulates:
    - Diurnal evapotranspiration (solar radiation cycle)
    - Gravity drainage above field capacity
    - Episodic rainfall events
    - Scheduled irrigation events
    - Inter-probe packing offsets and correlated sensor noise
    """

    def __init__(
        self,
        start_date: str = "2026-05-01T00:00:00Z",
        days: int = 90,
        interval_minutes: int = 15,
        field_capacity_vwc: float = 36.0,
        wilting_point_vwc: float = 10.0,
        drainage_rate: float = 0.04,
        max_et_rate_vwc_per_day: float = 3.5,
        base_noise_std: float = 0.25,
        random_seed: int = 42,
    ):
        self.start_date = pd.to_datetime(start_date, utc=True)
        self.days = days
        self.interval_minutes = interval_minutes
        self.steps_per_day = int(1440 / interval_minutes)
        self.total_steps = days * self.steps_per_day
        self.field_capacity = field_capacity_vwc
        self.wilting_point = wilting_point_vwc
        self.drainage_rate = drainage_rate
        self.max_et_rate = max_et_rate_vwc_per_day / self.steps_per_day
        self.base_noise_std = base_noise_std
        self.seed = random_seed

    def generate(self) -> Tuple[pd.DataFrame, List[WaterEvent]]:
        """Simulate continuous multi-sensor soil moisture and return DataFrame and logged water events."""
        rng = np.random.default_rng(self.seed)

        timestamps = [
            self.start_date + timedelta(minutes=i * self.interval_minutes)
            for i in range(self.total_steps)
        ]

        # Diurnal temperature and humidity cycle
        day_fractions = np.array([(t.hour + t.minute / 60.0) / 24.0 for t in timestamps])
        temp_cycle = 22.0 + 8.0 * np.sin(2 * np.pi * (day_fractions - 0.35)) + rng.normal(0, 0.5, self.total_steps)
        humidity_cycle = np.clip(
            70.0 - 25.0 * np.sin(2 * np.pi * (day_fractions - 0.35)) + rng.normal(0, 1.5, self.total_steps),
            20.0,
            98.0,
        )

        # Diurnal solar potential ET curve (active between 06:00 and 19:00, peak at 13:00)
        solar_et = np.maximum(0.0, np.sin(2 * np.pi * (day_fractions - 0.25)))

        # Generate schedule for rainfall and irrigation
        rain_mm = np.zeros(self.total_steps)
        irrigation_mm = np.zeros(self.total_steps)
        water_events: List[WaterEvent] = []

        # Rainfall events every 8-15 days
        rain_step = int(self.steps_per_day * rng.uniform(4, 7))
        while rain_step < self.total_steps - self.steps_per_day:
            duration_steps = rng.integers(4, 16)  # 1 to 4 hours of rain
            total_rain = rng.uniform(8.0, 30.0)
            rain_per_step = total_rain / duration_steps
            rain_mm[rain_step : rain_step + duration_steps] = rain_per_step

            water_events.append(
                WaterEvent(
                    timestamp=timestamps[rain_step],
                    event_type="rain",
                    amount_mm=float(np.round(total_rain, 1)),
                    notes=f"Natural rainfall duration {duration_steps * self.interval_minutes} mins",
                )
            )
            rain_step += int(self.steps_per_day * rng.uniform(8, 14))

        # Drip irrigation scheduled every 4-6 days (applied at 06:00)
        for day in range(3, self.days, 5):
            irrig_step = day * self.steps_per_day + int(6 * 60 / self.interval_minutes)
            if irrig_step < self.total_steps:
                irrig_amount = float(rng.uniform(10.0, 16.0))
                duration = 4  # 1 hour
                irrigation_mm[irrig_step : irrig_step + duration] = irrig_amount / duration
                water_events.append(
                    WaterEvent(
                        timestamp=timestamps[irrig_step],
                        event_type="irrigation",
                        amount_mm=float(np.round(irrig_amount, 1)),
                        notes=f"Scheduled drip irrigation {irrig_amount:.1f} mm",
                    )
                )

        # Infiltration conversion: 1 mm water addition produces approx +0.6% to +0.9% VWC in the top root zone
        infiltration_factor = 0.75
        water_input_vwc = (rain_mm + irrigation_mm) * infiltration_factor

        # Simulate true master soil moisture via differential water balance
        theta_true = np.zeros(self.total_steps)
        theta_current = 28.0  # Initial moisture in healthy root-zone range

        for t in range(self.total_steps):
            # Infiltration
            theta_current += water_input_vwc[t]

            # Gravity drainage if above field capacity
            if theta_current > self.field_capacity:
                drainage = (theta_current - self.field_capacity) * self.drainage_rate
                theta_current -= drainage

            # Evapotranspiration
            et_stress = np.clip(
                (theta_current - self.wilting_point) / (self.field_capacity - self.wilting_point),
                0.05,
                1.0,
            )
            et_loss = self.max_et_rate * solar_et[t] * et_stress
            theta_current = max(self.wilting_point, theta_current - et_loss)

            theta_true[t] = theta_current

        # Multi-probe generation: 4 colocated probes
        # Static installation packing offset for each probe
        probe_offsets = [-0.8, 0.4, -0.2, 0.6]
        # Infiltration sensitivity factor (micro-heterogeneity in water distribution)
        probe_infil_scale = [0.96, 1.02, 0.98, 1.04]

        sensor_data = {}
        for i in range(1, 5):
            idx = i - 1
            # Probes share master moisture with slight scaling and probe-specific offset
            sensor_signal = (
                theta_true * probe_infil_scale[idx]
                + probe_offsets[idx]
                + rng.normal(0, self.base_noise_std, self.total_steps)
            )
            sensor_data[f"sensor_{i}"] = np.clip(sensor_signal, 2.0, 55.0)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "sensor_1": sensor_data["sensor_1"],
                "sensor_2": sensor_data["sensor_2"],
                "sensor_3": sensor_data["sensor_3"],
                "sensor_4": sensor_data["sensor_4"],
                "rain_mm": np.round(rain_mm, 2),
                "irrigation_mm": np.round(irrigation_mm, 2),
                "ambient_temp": np.round(temp_cycle, 2),
                "relative_humidity": np.round(humidity_cycle, 1),
            }
        )

        validate_dataframe_schema(df)
        return df, water_events


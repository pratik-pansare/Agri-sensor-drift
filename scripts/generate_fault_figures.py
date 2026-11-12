"""Generate clean vs corrupted visualisations for all 5 fault types."""

import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import pandas as pd

from agri_drift.data.synthetic_generator import SyntheticHydrologyGenerator
from agri_drift.faults.injector import FaultInjector
from agri_drift.faults.visualizer import FaultVisualizer


def main():
    gen = SyntheticHydrologyGenerator(days=15, interval_minutes=15, random_seed=42)
    clean_df, _ = gen.generate()

    injector = FaultInjector(random_seed=42)
    demo_scenarios = [
        {"id": "drift_pos_2pct", "fault_type": "drift", "sensor_id": "sensor_2", "rate_vwc_per_day": 2.0, "start_hour": 48, "duration_hours": 72},
        {"id": "bias_pos_5pct", "fault_type": "bias", "sensor_id": "sensor_1", "offset_vwc": 6.0, "start_hour": 140, "duration_hours": 48},
        {"id": "stuck_freeze", "fault_type": "stuck_at", "sensor_id": "sensor_4", "mode": "freeze_current", "start_hour": 210, "duration_hours": 36},
        {"id": "transient_spikes", "fault_type": "spike", "sensor_id": "sensor_2", "amplitude_vwc": 18.0, "start_hour": 260, "duration_hours": 24},
        {"id": "high_noise", "fault_type": "noise", "sensor_id": "sensor_3", "std_multiplier": 5.0, "start_hour": 300, "duration_hours": 48},
    ]

    corrupted_df, manifest = injector.inject_benchmark_suite(clean_df, demo_scenarios)

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot for sensor_2 (contains drift and spikes)
    fig_path = out_dir / "figure3_fault_signatures_sensor2.png"
    FaultVisualizer.plot_fault_episode(
        df_clean=clean_df,
        df_corrupted=corrupted_df,
        manifest=manifest,
        sensor_id="sensor_2",
        output_path=fig_path,
        title="Injected Fault Signatures: Linear Drift and Transient Spikes (Sensor 2)",
    )
    print(f"Generated {fig_path}")


if __name__ == "__main__":
    main()


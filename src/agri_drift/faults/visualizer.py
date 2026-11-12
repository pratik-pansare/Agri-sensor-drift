"""Visualisation utilities for clean vs corrupted signals and fault ground truth."""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd

from agri_drift.faults.manifest import FaultManifest


class FaultVisualizer:
    """Renders high-resolution plots comparing healthy signals against injected fault signatures."""

    @staticmethod
    def plot_fault_episode(
        df_clean: pd.DataFrame,
        df_corrupted: pd.DataFrame,
        manifest: FaultManifest,
        sensor_id: str,
        output_path: Optional[str | Path] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot clean signal vs corrupted signal with fault onset and interval annotations."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

        times = pd.to_datetime(df_clean["timestamp"])

        # Top panel: Raw sensor traces
        ax1.plot(times, df_clean[sensor_id], label=f"{sensor_id} (Clean Ground Truth)", color="dodgerblue", alpha=0.8, linewidth=1.5)
        ax1.plot(times, df_corrupted[sensor_id], label=f"{sensor_id} (Corrupted/Faulty)", color="crimson", alpha=0.8, linestyle="--", linewidth=1.5)

        # Highlight fault intervals
        fault_colors = {
            "drift": "orange",
            "bias": "magenta",
            "stuck_at": "red",
            "spike": "purple",
            "noise": "brown",
        }

        for record in manifest.records:
            if record.target_sensor != sensor_id:
                continue
            t_start = pd.to_datetime(record.start_time)
            t_end = pd.to_datetime(record.end_time)
            c = fault_colors.get(record.fault_type, "red")

            ax1.axvspan(t_start, t_end, color=c, alpha=0.18, label=f"Fault: {record.fault_type.upper()}")
            ax1.axvline(t_start, color=c, linestyle=":", alpha=0.9)
            ax1.annotate(
                f"{record.fault_type.upper()}\n({record.fault_id})",
                xy=(t_start, df_corrupted[sensor_id].max() * 0.9),
                xytext=(10, 10),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=c, lw=1.2),
                fontsize=9,
                fontweight="bold",
                color=c,
            )

        ax1.set_ylabel("Soil Moisture (% VWC)", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        # Avoid duplicate legend labels
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True)

        if title:
            ax1.set_title(title, fontsize=13, fontweight="bold")
        else:
            ax1.set_title(f"Fault Injection Analysis: {sensor_id}", fontsize=13, fontweight="bold")

        # Bottom panel: Fault distortion delta
        delta = df_corrupted[sensor_id] - df_clean[sensor_id]
        ax2.plot(times, delta, color="firebrick", linewidth=1.2, label="Distortion Delta (Corrupted - Clean)")
        ax2.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
        ax2.set_ylabel(r"$\Delta$ VWC (%)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Time (UTC)", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True)

        plt.tight_layout()

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")

        return fig

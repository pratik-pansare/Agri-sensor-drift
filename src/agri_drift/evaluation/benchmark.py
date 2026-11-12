"""Comprehensive comparative benchmarking framework for sensor fault and drift detectors."""

import time
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from agri_drift.data.schema import HealthState, FaultType
from agri_drift.data.synthetic_generator import SyntheticHydrologyGenerator
from agri_drift.data.preprocessor import Preprocessor
from agri_drift.faults.injector import FaultInjector
from agri_drift.faults.manifest import FaultManifest
from agri_drift.detectors.threshold import ThresholdDetector
from agri_drift.detectors.neighbour import NeighbourResidualDetector
from agri_drift.detectors.cusum import CusumDetector
from agri_drift.detectors.page_hinkley import PageHinkleyDetector
from agri_drift.detectors.physics import PhysicsConsistencyDetector
from agri_drift.detectors.hybrid import HybridDecisionEngine
from agri_drift.evaluation.metrics import (
    compute_sample_classification_metrics,
    compute_detection_delay_and_far,
)


class BenchmarkRunner:
    """Executes systematic comparative research experiments across detectors and fault classes."""

    def __init__(self, config_path: str = "configs/experiment.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.seeds = self.config["experiment"].get("random_seeds", [42, 101, 777])
        self.metrics_dir = Path(self.config["output"].get("metrics_dir", "results/metrics"))
        self.figures_dir = Path(self.config["output"].get("figures_dir", "results/figures"))
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run_full_evaluation(self) -> Dict[str, pd.DataFrame]:
        """Run full multi-seed evaluation across all 4 detector systems and fault types."""
        all_overall_results = []
        all_fault_results = []

        sensors = ["sensor_1", "sensor_2", "sensor_3", "sensor_4"]

        # Define benchmark fault scenarios within the test window (~216 hours total)
        fault_scenarios = [
            {"id": "drift_pos", "fault_type": "drift", "sensor_id": "sensor_2", "rate_vwc_per_day": 1.5, "start_hour": 12, "duration_hours": 36},
            {"id": "drift_neg", "fault_type": "drift", "sensor_id": "sensor_3", "rate_vwc_per_day": -1.2, "start_hour": 54, "duration_hours": 36},
            {"id": "bias_step", "fault_type": "bias", "sensor_id": "sensor_1", "offset_vwc": 5.0, "start_hour": 96, "duration_hours": 24},
            {"id": "stuck_freeze", "fault_type": "stuck_at", "sensor_id": "sensor_4", "mode": "freeze_current", "start_hour": 126, "duration_hours": 24},
            {"id": "spikes_burst", "fault_type": "spike", "sensor_id": "sensor_2", "amplitude_vwc": 16.0, "start_hour": 156, "duration_hours": 18},
            {"id": "excess_noise", "fault_type": "noise", "sensor_id": "sensor_3", "std_multiplier": 4.5, "start_hour": 180, "duration_hours": 24},
        ]

        for seed in self.seeds:
            # 1. Generate clean dataset (60 days)
            gen = SyntheticHydrologyGenerator(days=60, interval_minutes=15, random_seed=seed)
            clean_df, events = gen.generate()

            # 2. Chronological split: 70% train, 15% val, 15% test
            train_df, val_df, test_df = Preprocessor.chronological_split(clean_df, 0.70, 0.15, 0.15)

            # 3. Inject benchmark fault scenarios strictly into test partition
            injector = FaultInjector(random_seed=seed, interval_minutes=15)
            corrupted_test_df, manifest = injector.inject_benchmark_suite(test_df, fault_scenarios)

            # 4. Train spatial models on healthy train partition
            neighbour_det = NeighbourResidualDetector(target_sensors=sensors)
            neighbour_det.fit(train_df)

            # 5. Evaluate the 4 systems on corrupted test set
            system_predictions = self._evaluate_systems_on_test(
                train_df=train_df,
                test_df=corrupted_test_df,
                events=events,
                sensors=sensors,
                neighbour_det=neighbour_det,
            )

            # 6. Compute metrics per system and per fault type
            for sys_name, pred_masks in system_predictions.items():
                # Aggregated across all sensors
                y_true_all = []
                y_pred_all = []
                delays = []
                fa_events = 0
                total_sensor_weeks = 0

                for s in sensors:
                    gt_mask = manifest.get_ground_truth_mask(corrupted_test_df, s)
                    p_mask = pred_masks[s]

                    y_true_all.extend(gt_mask.values)
                    y_pred_all.extend(p_mask.values)

                    delay_info = compute_detection_delay_and_far(
                        df=corrupted_test_df,
                        sensor_id=s,
                        ground_truth_mask=gt_mask,
                        prediction_mask=p_mask,
                        manifest_records=manifest.records,
                    )
                    if not np.isnan(delay_info["mean_detection_delay_hours"]):
                        delays.append(delay_info["mean_detection_delay_hours"])
                    fa_events += delay_info["false_alarm_events"]

                # 4 sensors * test duration in weeks
                test_duration_weeks = (len(test_df) * 15 / (60 * 24 * 7)) * len(sensors)
                far_rate = fa_events / test_duration_weeks if test_duration_weeks > 0 else 0.0

                cls_metrics = compute_sample_classification_metrics(np.array(y_true_all), np.array(y_pred_all))
                all_overall_results.append({
                    "seed": seed,
                    "system": sys_name,
                    "precision": cls_metrics["precision"],
                    "recall": cls_metrics["recall"],
                    "f1_score": cls_metrics["f1_score"],
                    "mean_detection_delay_hours": float(np.mean(delays)) if delays else np.nan,
                    "false_alarms_per_sensor_week": far_rate,
                })

                # Breakdown by fault type
                for ft in [FaultType.DRIFT.value, FaultType.BIAS.value, FaultType.STUCK_AT.value, FaultType.SPIKE.value, FaultType.NOISE.value]:
                    y_true_ft = []
                    y_pred_ft = []
                    for s in sensors:
                        gt_ft = manifest.get_ground_truth_mask(corrupted_test_df, s, fault_type=ft)
                        y_true_ft.extend(gt_ft.values)
                        y_pred_ft.extend(pred_masks[s].values)

                    ft_metrics = compute_sample_classification_metrics(np.array(y_true_ft), np.array(y_pred_ft))
                    all_fault_results.append({
                        "seed": seed,
                        "system": sys_name,
                        "fault_type": ft,
                        "precision": ft_metrics["precision"],
                        "recall": ft_metrics["recall"],
                        "f1_score": ft_metrics["f1_score"],
                    })

        # Aggregate metrics across seeds
        overall_df = pd.DataFrame(all_overall_results)
        summary_overall = overall_df.groupby("system").agg({
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1_score": ["mean", "std"],
            "mean_detection_delay_hours": ["mean", "std"],
            "false_alarms_per_sensor_week": ["mean", "std"],
        }).round(3)

        summary_overall.columns = [f"{c[0]}_{c[1]}" for c in summary_overall.columns]
        summary_overall = summary_overall.reset_index()

        fault_df = pd.DataFrame(all_fault_results)
        summary_fault = fault_df.groupby(["system", "fault_type"]).agg({
            "precision": "mean",
            "recall": "mean",
            "f1_score": "mean",
        }).round(3).reset_index()

        # Save to CSV
        overall_path = self.metrics_dir / "overall_comparison.csv"
        fault_path = self.metrics_dir / "metrics_by_fault_type.csv"
        summary_overall.to_csv(overall_path, index=False)
        summary_fault.to_csv(fault_path, index=False)

        # Generate publication figures
        self._generate_figures(corrupted_test_df, manifest, summary_overall, summary_fault)

        return {
            "overall": summary_overall,
            "by_fault": summary_fault,
        }

    def _evaluate_systems_on_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        events: List[Any],
        sensors: List[str],
        neighbour_det: NeighbourResidualDetector,
    ) -> Dict[str, Dict[str, pd.Series]]:
        """Run all 4 systems on the same test data stream and return boolean anomaly predictions."""
        n_steps = len(test_df)
        timestamps = pd.to_datetime(test_df["timestamp"], utc=True)

        # 1. Baseline Threshold Detector
        thresh_preds = {s: np.zeros(n_steps, dtype=bool) for s in sensors}
        thresh_det = ThresholdDetector(min_vwc=3.0, max_vwc=55.0, max_rate=8.0)
        for s in sensors:
            thresh_det.reset()
            for i, val in enumerate(test_df[s].values):
                alert = thresh_det.update(s, float(val))
                thresh_preds[s][i] = alert.is_anomaly

        # 2. Neighbour + CUSUM
        cusum_preds = {s: np.zeros(n_steps, dtype=bool) for s in sensors}
        cusum_dets = {s: CusumDetector(allowance_k=0.5, threshold_h=5.0) for s in sensors}
        for i in range(n_steps):
            row = {s: float(test_df.loc[i, s]) for s in sensors}
            for s in sensors:
                _, _, std_res = neighbour_det.predict(s, row)
                c_res = cusum_dets[s].update(std_res if not np.isnan(std_res) else 0.0)
                cusum_preds[s][i] = c_res.is_change

        # 3. Neighbour + Page-Hinkley
        ph_preds = {s: np.zeros(n_steps, dtype=bool) for s in sensors}
        ph_dets = {s: PageHinkleyDetector(delta=0.5, threshold_lambda=15.0) for s in sensors}
        for i in range(n_steps):
            row = {s: float(test_df.loc[i, s]) for s in sensors}
            for s in sensors:
                _, _, std_res = neighbour_det.predict(s, row)
                p_res = ph_dets[s].update(std_res if not np.isnan(std_res) else 0.0)
                ph_preds[s][i] = p_res.is_change

        # 4. Hybrid Physics-Aware Engine
        hybrid_preds = {s: np.zeros(n_steps, dtype=bool) for s in sensors}
        physics_det = PhysicsConsistencyDetector(target_sensors=sensors)
        # Register relevant water events
        for ev in events:
            ev_ts = pd.to_datetime(ev.timestamp, utc=True)
            if ev_ts >= timestamps.iloc[0] and ev_ts <= timestamps.iloc[-1]:
                # find closest index
                closest_idx = int(np.argmin(np.abs((timestamps - ev_ts).dt.total_seconds())))
                base_readings = {s: float(test_df.loc[closest_idx, s]) for s in sensors}
                physics_det.register_event(ev, base_readings)

        hybrid_engine = HybridDecisionEngine(
            target_sensors=sensors,
            threshold_detector=ThresholdDetector(min_vwc=3.0, max_vwc=55.0, max_rate=8.0),
            neighbour_detector=neighbour_det,
            physics_detector=physics_det,
            confirmation_window_steps=2,
        )

        for i in range(n_steps):
            t = timestamps.iloc[i]
            row = {s: float(test_df.loc[i, s]) for s in sensors}
            decisions = hybrid_engine.process_step(t, row)
            for s in sensors:
                is_flagged = decisions[s].state in [HealthState.SUSPECTED_FAULT, HealthState.CONFIRMED_FAULT]
                hybrid_preds[s][i] = is_flagged

        return {
            "Baseline-Threshold": {s: pd.Series(thresh_preds[s], index=test_df.index) for s in sensors},
            "Neighbour-CUSUM": {s: pd.Series(cusum_preds[s], index=test_df.index) for s in sensors},
            "Neighbour-PageHinkley": {s: pd.Series(ph_preds[s], index=test_df.index) for s in sensors},
            "Hybrid-Physics-Aware": {s: pd.Series(hybrid_preds[s], index=test_df.index) for s in sensors},
        }

    def _generate_figures(
        self,
        test_df: pd.DataFrame,
        manifest: FaultManifest,
        overall_summary: pd.DataFrame,
        fault_summary: pd.DataFrame,
    ) -> None:
        """Render publication-quality figures."""
        # Figure 1: Overall Model Comparison Bar Chart
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        models = overall_summary["system"].tolist()
        x = np.arange(len(models))
        width = 0.25

        ax1.bar(x - width, overall_summary["precision_mean"], width, label="Precision", color="#2b5c8f")
        ax1.bar(x, overall_summary["recall_mean"], width, label="Recall", color="#e27c38")
        ax1.bar(x + width, overall_summary["f1_score_mean"], width, label="F1-Score", color="#2a9d8f")

        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=15, ha="right", fontweight="bold")
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel("Score", fontweight="bold")
        ax1.set_title("Overall Detector Performance Across All Fault Classes", fontweight="bold")
        ax1.legend(loc="lower right")
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Subplot 2: Detection Delay & False Alarm Rate
        ax2_twin = ax2.twinx()
        ax2.bar(x - 0.15, overall_summary["mean_detection_delay_hours_mean"], 0.3, label="Delay (Hours)", color="#e76f51")
        ax2_twin.bar(x + 0.15, overall_summary["false_alarms_per_sensor_week_mean"], 0.3, label="False Alarms/Week", color="#457b9d", hatch="//")

        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=15, ha="right", fontweight="bold")
        ax2.set_ylabel("Mean Detection Delay (Hours)", color="#e76f51", fontweight="bold")
        ax2_twin.set_ylabel("False Alarms / Sensor-Week", color="#457b9d", fontweight="bold")
        ax2.set_title("Operational Edge Tradeoffs: Delay vs. False Alarms", fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        fig1.savefig(self.figures_dir / "figure1_model_comparison.png", dpi=300)
        fig1.savefig(self.figures_dir / "figure1_model_comparison.pdf")
        plt.close(fig1)

        # Figure 2: Performance by Fault Type (F1-Score Heatmap / Grouped Bar)
        fig2, ax = plt.subplots(figsize=(12, 6))
        pivot_f1 = fault_summary.pivot(index="fault_type", columns="system", values="f1_score")
        pivot_f1.plot(kind="bar", ax=ax, colormap="tab10", width=0.8)
        ax.set_ylabel("F1-Score", fontweight="bold", fontsize=11)
        ax.set_xlabel("Fault Type", fontweight="bold", fontsize=11)
        ax.set_title("Detector F1-Score Breakdown by Fault Class", fontweight="bold", fontsize=13)
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.xticks(rotation=0, fontweight="bold")
        plt.legend(title="Detector System", loc="lower right")
        plt.tight_layout()
        fig2.savefig(self.figures_dir / "figure2_f1_by_fault_type.png", dpi=300)
        fig2.savefig(self.figures_dir / "figure2_f1_by_fault_type.pdf")
        plt.close(fig2)

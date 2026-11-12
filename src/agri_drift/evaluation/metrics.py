"""Performance evaluation metrics: Precision, Recall, F1, Detection Delay, and False Alarm Rate."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def compute_sample_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute sample-level binary classification metrics: Precision, Recall, F1, Specificity, FPR."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum((~y_true) & y_pred))
    fn = int(np.sum(y_true & (~y_pred)))
    tn = int(np.sum((~y_true) & (~y_pred)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
    }


def compute_detection_delay_and_far(
    df: pd.DataFrame,
    sensor_id: str,
    ground_truth_mask: pd.Series,
    prediction_mask: pd.Series,
    manifest_records: List[Any],
    sampling_interval_minutes: int = 15,
) -> Dict[str, float]:
    """Compute Mean Detection Delay (hours) and False Alarms per Sensor-Week."""
    delays_hours = []
    timestamps = pd.to_datetime(df["timestamp"], utc=True)

    # Calculate detection delay for each injected fault episode
    for rec in manifest_records:
        if rec.target_sensor != sensor_id:
            continue

        r_start = pd.to_datetime(rec.start_time, utc=True)
        r_end = pd.to_datetime(rec.end_time, utc=True)

        episode_mask = (timestamps >= r_start) & (timestamps <= r_end)
        detected_indices = np.where(episode_mask & prediction_mask)[0]

        if len(detected_indices) > 0:
            first_det_time = timestamps.iloc[detected_indices[0]]
            delay = (first_det_time - r_start).total_seconds() / 3600.0
            delays_hours.append(max(0.0, delay))

    mean_delay = float(np.mean(delays_hours)) if delays_hours else np.nan

    # Calculate False Alarm Rate (FAR) on healthy periods
    healthy_mask = ~ground_truth_mask
    healthy_predictions = prediction_mask[healthy_mask]

    # Count distinct false alarm events (consecutive positive runs)
    diffs = np.diff(healthy_predictions.astype(int), prepend=0)
    fa_event_count = int(np.sum(diffs == 1))

    total_healthy_steps = int(np.sum(healthy_mask))
    total_healthy_hours = (total_healthy_steps * sampling_interval_minutes) / 60.0
    total_sensor_weeks = total_healthy_hours / (24.0 * 7.0)

    far_per_week = float(fa_event_count / total_sensor_weeks) if total_sensor_weeks > 0 else 0.0

    return {
        "mean_detection_delay_hours": mean_delay,
        "detected_episodes": len(delays_hours),
        "false_alarm_events": fa_event_count,
        "false_alarms_per_sensor_week": far_per_week,
    }

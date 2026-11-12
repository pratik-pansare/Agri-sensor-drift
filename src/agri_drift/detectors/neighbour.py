"""Neighbour-residual predictor with contamination defense using healthy historical training."""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class NeighbourResidualDetector:
    """Predicts a target sensor's measurement from its spatial neighbours.

    Trains OLS regression models strictly on healthy data.
    Provides dynamic contamination defense: if a reference sensor is currently flagged
    as faulty, the detector seamlessly falls back to a sub-model trained without that sensor.
    """

    def __init__(self, target_sensors: List[str]):
        self.target_sensors = target_sensors
        # Models dictionary: target -> frozenset(available_refs) -> LinearRegression
        self.models: Dict[str, Dict[frozenset, LinearRegression]] = {}
        # Baseline residual statistics on healthy training data: target -> (mean, std)
        self.stats: Dict[str, Tuple[float, float]] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit neighbour prediction models for each target sensor across all reference subsets."""
        for target in self.target_sensors:
            all_refs = [s for s in self.target_sensors if s != target]
            self.models[target] = {}

            # Fit primary model with all available reference sensors
            X_full = train_df[all_refs].values
            y = train_df[target].values

            primary_model = LinearRegression()
            primary_model.fit(X_full, y)
            self.models[target][frozenset(all_refs)] = primary_model

            # Calculate baseline residual mean and standard deviation
            y_pred = primary_model.predict(X_full)
            residuals = y - y_pred
            mu_0 = float(np.mean(residuals))
            sigma_0 = float(np.std(residuals))
            if sigma_0 < 1e-4:
                sigma_0 = 1.0  # Guard against division by zero
            self.stats[target] = (mu_0, sigma_0)

            # Fit leave-one-out models for contamination isolation
            if len(all_refs) > 1:
                for excluded in all_refs:
                    sub_refs = [s for s in all_refs if s != excluded]
                    sub_model = LinearRegression()
                    sub_model.fit(train_df[sub_refs].values, y)
                    self.models[target][frozenset(sub_refs)] = sub_model

    def predict(
        self,
        target_sensor: str,
        readings: Dict[str, float],
        excluded_sensors: Optional[Set[str]] = None,
    ) -> Tuple[float, float, float]:
        """Predict expected reading, compute raw residual and standardized residual.

        Returns: (predicted_value, raw_residual, standardized_residual)
        """
        if target_sensor not in self.models:
            raise KeyError(f"Target sensor '{target_sensor}' not fitted.")

        actual_val = readings[target_sensor]
        if np.isnan(actual_val):
            return np.nan, np.nan, np.nan

        all_refs = [s for s in self.target_sensors if s != target_sensor]
        excluded = set(excluded_sensors or set())

        # Select clean reference sensors
        valid_refs = [s for s in all_refs if s not in excluded and not np.isnan(readings.get(s, np.nan))]

        if not valid_refs:
            # Fallback when no clean references remain
            mu_0, sigma_0 = self.stats[target_sensor]
            return actual_val, 0.0, 0.0

        ref_key = frozenset(valid_refs)

        # Retrieve exact model or best matching subset model
        if ref_key in self.models[target_sensor]:
            model = self.models[target_sensor][ref_key]
            features = np.array([[readings[s] for s in valid_refs]])
            predicted_val = float(model.predict(features)[0])
        else:
            # Fallback to mean of valid reference sensors adjusted for offset
            predicted_val = float(np.mean([readings[s] for s in valid_refs]))

        raw_residual = actual_val - predicted_val
        mu_0, sigma_0 = self.stats[target_sensor]
        std_residual = (raw_residual - mu_0) / sigma_0

        return predicted_val, raw_residual, std_residual

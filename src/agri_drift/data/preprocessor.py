"""Time-series preprocessing, alignment, unit normalization, and leakage-free splitting."""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd


class Preprocessor:
    """Preprocesses agricultural time-series data for edge drift detection."""

    def __init__(
        self,
        resample_interval: str = "15min",
        max_interp_gap_steps: int = 4,
        min_vwc: float = 0.0,
        max_vwc: float = 100.0,
    ):
        self.resample_interval = resample_interval
        self.max_interp_gap_steps = max_interp_gap_steps
        self.min_vwc = min_vwc
        self.max_vwc = max_vwc

    def align_and_resample(
        self,
        df: pd.DataFrame,
        sensor_cols: Optional[List[str]] = None,
        event_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Resample time-series to a regular temporal grid, handling sensor and event columns appropriately."""
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if sensor_cols is None:
            sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        if event_cols is None:
            event_cols = [c for c in df.columns if c in ["rain_mm", "irrigation_mm"]]

        df = df.set_index("timestamp")

        # Aggregation rules: sensors by mean, rainfall/irrigation by sum, weather by mean
        agg_dict = {}
        for col in df.columns:
            if col in event_cols:
                agg_dict[col] = "sum"
            else:
                agg_dict[col] = "mean"

        resampled = df.resample(self.resample_interval).agg(agg_dict)

        # Interpolate bounded short gaps for sensor channels
        for col in sensor_cols:
            resampled[col] = resampled[col].interpolate(
                method="time",
                limit=self.max_interp_gap_steps,
                limit_direction="inside",
            )

        # For event columns (rain/irrigation), fill missing with 0.0
        for col in event_cols:
            if col in resampled.columns:
                resampled[col] = resampled[col].fillna(0.0)

        resampled = resampled.reset_index()
        return resampled

    def normalize_units(self, df: pd.DataFrame, sensor_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Ensure sensor values are expressed in percentage Volumetric Water Content (% VWC, 0-100)."""
        df = df.copy()
        if sensor_cols is None:
            sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

        for col in sensor_cols:
            max_val = df[col].max()
            # If data is in ratio m^3/m^3 (e.g. max_val <= 1.0), convert to percentage
            if max_val is not None and not np.isnan(max_val) and max_val <= 1.0:
                df[col] = df[col] * 100.0

            # Clamp physical limits
            df[col] = df[col].clip(lower=self.min_vwc, upper=self.max_vwc)

        return df

    @staticmethod
    def chronological_split(
        df: pd.DataFrame,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split time-series chronologically into Train, Validation, and Test sets without future lookahead."""
        total_frac = train_frac + val_frac + test_frac
        if not np.isclose(total_frac, 1.0):
            raise ValueError(f"Fractions must sum to 1.0, got {total_frac}")

        df = df.sort_values("timestamp").reset_index(drop=True)
        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)

        return train_df, val_df, test_df

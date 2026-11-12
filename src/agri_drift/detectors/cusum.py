"""Tabular Cumulative Sum (CUSUM) change-point detector for drift monitoring."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CusumResult:
    """Result of a CUSUM step update."""
    is_change: bool
    direction: Optional[str] = None  # "positive", "negative", or None
    s_pos: float = 0.0
    s_neg: float = 0.0
    threshold: float = 5.0
    allowance: float = 0.5


class CusumDetector:
    """Two-sided tabular CUSUM algorithm for detecting small, persistent shifts and drift in residuals.

    Positive CUSUM: S_t^+ = max(0, S_{t-1}^+ + z_t - k)
    Negative CUSUM: S_t^- = max(0, S_{t-1}^- - z_t - k)

    where z_t is the standardized residual, k is the allowance parameter,
    and h is the decision boundary.
    """

    def __init__(
        self,
        allowance_k: float = 0.5,
        threshold_h: float = 5.0,
        reset_on_alert: bool = True,
    ):
        self.k = allowance_k
        self.h = threshold_h
        self.reset_on_alert = reset_on_alert
        self.s_pos: float = 0.0
        self.s_neg: float = 0.0
        self.total_steps: int = 0

    def reset(self) -> None:
        """Reset internal accumulator statistics."""
        self.s_pos = 0.0
        self.s_neg = 0.0

    def update(self, z: float) -> CusumResult:
        """Process standardized residual value z and evaluate change boundaries."""
        self.total_steps += 1

        self.s_pos = max(0.0, self.s_pos + z - self.k)
        self.s_neg = max(0.0, self.s_neg - z - self.k)

        is_positive = self.s_pos > self.h
        is_negative = self.s_neg > self.h
        is_change = is_positive or is_negative

        direction = None
        if is_positive and is_negative:
            direction = "bidirectional"
        elif is_positive:
            direction = "positive"
        elif is_negative:
            direction = "negative"

        result = CusumResult(
            is_change=is_change,
            direction=direction,
            s_pos=self.s_pos,
            s_neg=self.s_neg,
            threshold=self.h,
            allowance=self.k,
        )

        if is_change and self.reset_on_alert:
            self.reset()

        return result

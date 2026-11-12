"""Page-Hinkley change detection test for sequential residual monitoring."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PageHinkleyResult:
    """Result of a Page-Hinkley step update."""
    is_change: bool
    direction: Optional[str] = None
    ph_stat: float = 0.0
    threshold: float = 15.0


class PageHinkleyDetector:
    """Sequential Page-Hinkley test for detecting abrupt and gradual changes in streaming residuals.

    Tracks cumulative deviations from running mean with configurable slack (delta)
    and threshold (lambda).
    """

    def __init__(
        self,
        delta: float = 0.5,
        threshold_lambda: float = 15.0,
        alpha: float = 0.99,
        reset_on_alert: bool = True,
    ):
        self.delta = delta
        self.threshold = threshold_lambda
        self.alpha = alpha
        self.reset_on_alert = reset_on_alert

        self.mean: float = 0.0
        self.n: int = 0
        self.sum_pos: float = 0.0
        self.min_sum_pos: float = 0.0
        self.sum_neg: float = 0.0
        self.max_sum_neg: float = 0.0

    def reset(self) -> None:
        """Reset internal accumulator statistics."""
        self.mean = 0.0
        self.n = 0
        self.sum_pos = 0.0
        self.min_sum_pos = 0.0
        self.sum_neg = 0.0
        self.max_sum_neg = 0.0

    def update(self, val: float) -> PageHinkleyResult:
        """Process value and update Page-Hinkley test statistics."""
        self.n += 1

        # Running average with exponential forgetting or simple mean
        if self.n == 1:
            self.mean = val
        else:
            self.mean = self.alpha * self.mean + (1.0 - self.alpha) * val

        # Positive shift tracking
        self.sum_pos += (val - self.mean - self.delta)
        if self.sum_pos < self.min_sum_pos:
            self.min_sum_pos = self.sum_pos
        ph_pos = self.sum_pos - self.min_sum_pos

        # Negative shift tracking
        self.sum_neg += (val - self.mean + self.delta)
        if self.sum_neg > self.max_sum_neg:
            self.max_sum_neg = self.sum_neg
        ph_neg = self.max_sum_neg - self.sum_neg

        is_pos_change = ph_pos > self.threshold
        is_neg_change = ph_neg > self.threshold
        is_change = is_pos_change or is_neg_change

        direction = None
        max_stat = max(ph_pos, ph_neg)
        if is_pos_change and is_neg_change:
            direction = "bidirectional"
        elif is_pos_change:
            direction = "positive"
        elif is_neg_change:
            direction = "negative"

        result = PageHinkleyResult(
            is_change=is_change,
            direction=direction,
            ph_stat=max_stat,
            threshold=self.threshold,
        )

        if is_change and self.reset_on_alert:
            self.reset()

        return result

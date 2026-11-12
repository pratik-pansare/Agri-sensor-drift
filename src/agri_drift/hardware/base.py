"""Abstract base class for hardware and mock sensor reading adapters."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class SensorReader(ABC):
    """Abstract interface for analog moisture probes and environmental sensors."""

    @abstractmethod
    def read_channels(self) -> Dict[str, Tuple[float, float]]:
        """Read all configured moisture channels.

        Returns:
            Dictionary mapping sensor_id -> (raw_voltage, calibrated_vwc_percent)
        """
        pass

    @abstractmethod
    def read_context(self) -> Dict[str, Optional[float]]:
        """Read environmental context variables (temperature, relative humidity).

        Returns:
            Dictionary with 'ambient_temp' and 'relative_humidity'
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check whether sensor hardware bus is reachable."""
        pass

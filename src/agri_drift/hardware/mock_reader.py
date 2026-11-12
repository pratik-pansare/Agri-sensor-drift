"""Mock hardware adapter simulating ADS1115 ADC and soil moisture probes for laptop validation."""

from typing import Dict, Tuple, Optional
import numpy as np

from agri_drift.hardware.base import SensorReader


class MockSensorReader(SensorReader):
    """Simulates ADS1115 16-bit ADC voltage conversions and calibrated moisture values.

    Allows injecting mock hardware faults (disconnection, voltage rail saturation, noise)
    without physical Raspberry Pi hardware.
    """

    def __init__(
        self,
        channels: Optional[Dict[str, int]] = None,
        random_seed: int = 42,
    ):
        self.channels = channels or {
            "sensor_1": 0,
            "sensor_2": 1,
            "sensor_3": 2,
            "sensor_4": 3,
        }
        self.rng = np.random.default_rng(random_seed)
        self.disconnected_sensors: set = set()
        self.simulated_vwc: Dict[str, float] = {s: 26.0 for s in self.channels}

    def set_simulated_vwc(self, sensor_id: str, vwc: float) -> None:
        """Set the underlying simulated soil moisture."""
        self.simulated_vwc[sensor_id] = vwc

    def simulate_disconnect(self, sensor_id: str) -> None:
        """Simulate wire disconnection or open circuit."""
        self.disconnected_sensors.add(sensor_id)

    def simulate_reconnect(self, sensor_id: str) -> None:
        """Restore sensor connection."""
        self.disconnected_sensors.discard(sensor_id)

    def is_connected(self) -> bool:
        return True

    def read_channels(self) -> Dict[str, Tuple[float, float]]:
        """Return simulated raw voltages and converted VWC percentages.

        Conversion curve: V_air = 2.8V (0% VWC), V_water = 1.2V (55% VWC).
        V = 2.8 - (vwc / 55.0) * 1.6
        """
        results = {}
        for s in self.channels:
            if s in self.disconnected_sensors:
                results[s] = (np.nan, np.nan)
                continue

            base_vwc = self.simulated_vwc.get(s, 25.0)
            noise_vwc = base_vwc + float(self.rng.normal(0, 0.2))
            clamped_vwc = float(np.clip(noise_vwc, 0.0, 100.0))

            # Simulate ADC voltage (inverted response typical of capacitive v1.2 probes)
            voltage = 2.8 - (clamped_vwc / 55.0) * 1.6 + float(self.rng.normal(0, 0.005))
            results[s] = (round(voltage, 4), round(clamped_vwc, 2))

        return results

    def read_context(self) -> Dict[str, Optional[float]]:
        """Simulate DHT22 temperature and humidity reading."""
        temp = 24.5 + float(self.rng.normal(0, 0.3))
        humidity = 62.0 + float(self.rng.normal(0, 1.0))
        return {
            "ambient_temp": round(temp, 2),
            "relative_humidity": round(humidity, 1),
        }


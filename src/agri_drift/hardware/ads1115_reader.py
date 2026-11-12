"""Hardware adapter for ADS1115 16-bit ADC over I2C on Raspberry Pi."""

import logging
from typing import Dict, Tuple, Optional
import numpy as np

from agri_drift.hardware.base import SensorReader

logger = logging.getLogger("agri_drift.hardware.ads1115")


class ADS1115SensorReader(SensorReader):
    """Interacts with ADS1115 16-bit ADC over I2C to read 4 analog moisture probes.

    Gracefully falls back when executed on non-Pi laptop environments where
    `board` or `adafruit_ads1x15` are unavailable.
    """

    def __init__(
        self,
        i2c_address: int = 0x48,
        gain: int = 1,
        channel_mapping: Optional[Dict[str, int]] = None,
        dry_voltages: Optional[Dict[str, float]] = None,
        wet_voltages: Optional[Dict[str, float]] = None,
    ):
        self.i2c_address = i2c_address
        self.gain = gain
        self.channel_mapping = channel_mapping or {
            "sensor_1": 0,
            "sensor_2": 1,
            "sensor_3": 2,
            "sensor_4": 3,
        }
        # In-air (dry) calibration voltages (typically ~2.8V - 3.0V)
        self.dry_voltages = dry_voltages or {s: 2.85 for s in self.channel_mapping}
        # In-water (wet) calibration voltages (typically ~1.2V - 1.4V)
        self.wet_voltages = wet_voltages or {s: 1.25 for s in self.channel_mapping}

        self.ads = None
        self._init_hardware()

    def _init_hardware(self) -> None:
        """Attempt to initialize CircuitPython I2C bus and ADS1115 ADC."""
        try:
            import board
            import busio
            import adafruit_ads1x15.ads1115 as ADS
            from adafruit_ads1x15.analog_in import AnalogIn

            i2c = busio.I2C(board.SCL, board.SDA)
            self.ads = ADS.ADS1115(i2c, address=self.i2c_address, gain=self.gain)
            self.analog_ins = {
                s: AnalogIn(self.ads, getattr(ADS, f"P{ch}"))
                for s, ch in self.channel_mapping.items()
            }
            logger.info("ADS1115 ADC successfully initialized over I2C.")
        except (ImportError, NotImplementedError, AttributeError, Exception) as e:
            logger.warning(
                f"ADS1115 hardware initialization bypassed (normal on non-Pi development laptop): {e}"
            )
            self.ads = None
            self.analog_ins = {}

    def is_connected(self) -> bool:
        return self.ads is not None

    def read_channels(self) -> Dict[str, Tuple[float, float]]:
        """Read 16-bit analog voltages and apply linear calibration to % VWC."""
        if not self.is_connected():
            raise RuntimeError(
                "Physical ADS1115 hardware is not connected. Use MockSensorReader on development laptops."
            )

        results = {}
        for s, analog_in in self.analog_ins.items():
            try:
                voltage = analog_in.voltage
                # Two-point calibration curve
                v_dry = self.dry_voltages[s]
                v_wet = self.wet_voltages[s]

                # Capacitive probes decrease voltage with higher moisture
                vwc = ((v_dry - voltage) / (v_dry - v_wet)) * 55.0
                clamped_vwc = float(np.clip(vwc, 0.0, 100.0))
                results[s] = (round(voltage, 4), round(clamped_vwc, 2))
            except Exception as e:
                logger.error(f"Error reading channel for {s}: {e}")
                results[s] = (np.nan, np.nan)

        return results

    def read_context(self) -> Dict[str, Optional[float]]:
        """Read DHT22 context sensor if connected."""
        # Optional DHT22 integration on Pi GPIO
        return {"ambient_temp": None, "relative_humidity": None}

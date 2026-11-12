#!/usr/bin/env bash
# Agri-Drift Raspberry Pi Setup & Installation Script

set -e

echo "=== [1/5] Updating Raspberry Pi OS package lists ==="
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv i2c-tools git libatlas-base-dev

echo "=== [2/5] Enabling I2C Interface ==="
sudo raspi-config nonint do_i2c 0
echo "Verifying I2C bus device nodes..."
ls -l /dev/i2c* || true

echo "=== [3/5] Setting up Python virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install adafruit-circuitpython-ads1x15

echo "=== [4/5] Testing ADS1115 I2C connectivity ==="
echo "Scanning I2C bus 1 for ADS1115 (default address 0x48)..."
i2cdetect -y 1 || true

echo "=== [5/5] Installing systemd Edge Monitoring Service ==="
sudo cp deployment/agri_drift.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable agri_drift.service

echo "=========================================================="
echo "Agri-Drift setup complete!"
echo "To start the edge logging service:"
echo "  sudo systemctl start agri_drift.service"
echo "To view live diagnostic logs:"
echo "  sudo journalctl -u agri_drift.service -f"
echo "=========================================================="

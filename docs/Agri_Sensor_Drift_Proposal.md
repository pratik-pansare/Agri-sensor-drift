# Agri_Sensor_Drift_Proposal: Specification Document

## Sensor-Fault and Drift Detection in Agricultural IoT

### 1. Problem and Research Question
**Problem**: Low-cost soil sensors corrode, lose calibration and get stuck, and a slow drift looks a lot like the soil really drying out. A farmer who irrigates on a faulty reading wastes water or loses yield. Most deployments only check whether readings fall outside a fixed range, which misses slow drift completely.

**Research Question**: Can an edge node tell a real environmental change apart from sensor degradation by checking each reading against neighbouring sensors, physical relationships (rain or irrigation should raise moisture) and its own recent behaviour?

### 2. Scope
* Use a public soil-moisture time series as the "healthy" baseline and inject labelled faults into it: drift, bias offset, stuck-at value, spikes and noise.
* Add a small bench testbed: 3–4 soil-moisture sensors in pots of the same soil, one deliberately degraded (e.g., a corroded resistive probe or a capacitive probe with damaged coating), plus controlled watering events.
* Soil moisture is the main variable. pH and NPK probes are optional extras if the lab already has them.
* A Raspberry Pi is the sensing and edge node. The Pi has no analog inputs, so sensors connect through an ADS1115 ADC over I2C. It uses Wi-Fi to start, with a LoRa HAT as an option for field range.

### 3. Method
The detector flags a reading as suspect when the sensor disagrees with its neighbours and with physical expectations, rather than when it leaves a fixed range.
* **Baseline data**: Download a public multi-sensor soil-moisture series (e.g., an International Soil Moisture Network station with several depths and a rain gauge, or the Mendeley low-cost sensor calibration dataset) or synthetic hydrology.
* **Fault injection**: A script adds labelled faults at random times: linear drift (e.g., 0.5–2% VWC per day), constant bias, stuck-at, spike bursts and growing noise. This gives exact ground truth.
* **Detectors, simplest first**:
  1. Fixed range and rate-of-change thresholds (baseline).
  2. Neighbour residual: predict each sensor from the others (linear regression) and flag when the residual’s CUSUM or Page–Hinkley statistic crosses a limit. Drift shows up as a residual trend that neighbours do not share.
  3. Physics check: after rain or irrigation, moisture should rise within a set window; a sensor that does not respond is flagged.
  4. Optional: an isolation-forest or small LSTM model on the residuals.
* **Decision rule**: All sensors change together $\rightarrow$ real environmental change; one sensor diverges $\rightarrow$ suspected fault, labelled by type (drift, stuck, spike).
* **Edge detector on the Pi**: Run the residual and CUSUM logic as a Python service on the Raspberry Pi, so every reading is stored with a health flag. Alerts go to a dashboard or Telegram bot as plain farmer-facing messages.

### 4. Hardware and Software Specifications
* **Raspberry Pi 4 / 5**: Edge node running sensing service and detector.
* **ADS1115 16-bit ADC module**: 4 analog channels connected via I2C.
* **3 Capacitive soil-moisture sensors (v1.2)**: Healthy sensors.
* **1 Resistive soil-moisture probe**: Deliberately degraded sensor.
* **DHT22 temperature/humidity sensor**: Context variable for environmental physics check.
* **Software**: Python (pandas, scikit-learn, river/CUSUM), Raspberry Pi OS, Adafruit CircuitPython ADS1x15 library, systemd service, SQLite / Streamlit dashboard.

### 5. Work Plan
* Baseline dataset ingest & audit; bench testbed setup.
* Fault-injection engine & labelled benchmark dataset.
* Neighbour-residual + CUSUM / Page–Hinkley detectors.
* Hydrological physics check (rain/irrigation response).
* Comparative evaluation across all fault types and baselines.
* Live replay streamer & Streamlit alert dashboard.
* Raspberry Pi integration & hardware bench protocol.
* Final documentation, reproducibility scripts, and research report.

### 6. Deliverables & Evaluation Metrics
* Openly released fault-injection tool and labelled benchmark dataset.
* Modular Python package (`agri_drift`).
* Self-diagnosing edge service attaching health states to readings.
* Plain-language farmer alerts.
* Evaluation metrics: Precision, Recall, F1 per fault type; Mean Detection Delay (hours); False Alarms per sensor-week.


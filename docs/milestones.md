# Project Milestones (M0 – M8) Roadmap

This document outlines the sequential development lifecycle, inputs, implementation targets, automated verification criteria, and transition gates for the **Sensor-Fault and Drift Detection in Agricultural IoT** project.

---

### Milestone Overview Matrix

| Milestone | Title | Execution Environment | Primary Deliverable | Acceptance Gate |
| :--- | :--- | :--- | :--- | :--- |
| **M0** | Repo & Environment Setup | Laptop | Virtualenv, pyproject.toml, package structure, configs | Environment verified, `pytest` executes |
| **M1** | Dataset Ingestion & Preprocessing | Laptop | Data loaders, schema, dataset audit, synthetic baseline | Clean, aligned, multi-sensor time-series with splits |
| **M2** | Labelled Fault Injection | Laptop | Configurable 5-fault engine, parameter sweeps, manifests | Reproducible corrupted signals & ground-truth labels |
| **M3** | Detection Algorithms | Laptop | Threshold, Neighbour OLS, CUSUM, Page-Hinkley | Detectors run streaming & offline; unit tests pass |
| **M4** | Physics-Aware Hybrid Fusion | Laptop | Hydrological check + state machine (5 health states) | Interpretable health diagnosis & subtype alerts |
| **M5** | Evaluation & Benchmarking | Laptop | Metric calculations, PR curves, CSV reports, figures | Benchmark table, MDD, FAR, no test-set leakage |
| **M6** | Replay & Streamlit Dashboard | Laptop | Chronological streamer, SQLite, interactive UI | Live dashboard with SIMULATION badge & alerts |
| **M7** | Hardware Abstraction & Pi Prep | Laptop + Mock / (Optional Pi) | ADS1115 I2C adapter, mock driver, systemd service | Mock tests pass; hardware guidelines complete |
| **M8** | Documentation & Reproducibility | Laptop | Final report, user guide, verification package | One-command run script reproducing results |

---

### Detailed Milestone Specifications

#### Milestone M0: Repository and Development Environment
* **Objective**: Establish a clean, reproducible, modern Python package under `src/agri_drift/` with configuration management, dependency locks, and testing harness.
* **Required Inputs**: Master prompt specification and system configuration files.
* **Source Implementation**:
  - `pyproject.toml` / `setup.cfg` defining `agri_drift` package.
  - Core directory hierarchy (`configs/`, `data/`, `src/agri_drift/`, `tests/`, `results/`, `docs/`).
  - Base configuration files in `configs/default.yaml`.
* **Verification Criteria**:
  - `pip install -e .` succeeds in Python 3.11 virtual environment.
  - `pytest tests/` runs and discovers base tests without import errors.
  - No hardware-specific packages (e.g., Adafruit ADS1x15) required at import time on non-Pi platforms.

#### Milestone M1: Dataset Ingestion, Audit, and Preprocessing
* **Objective**: Ingest real and synthetic soil moisture time-series, evaluate candidates (ISMN and Mendeley), implement unified schema, clean/resample data, and partition chronologically.
* **Required Inputs**: Candidate datasets / synthetic hydrology specifications.
* **Source Implementation**:
  - `src/agri_drift/data/schema.py`: Pydantic/dataclass schema for sensor streams and environmental events.
  - `src/agri_drift/data/loader.py`: CSV and Parquet loaders.
  - `src/agri_drift/data/preprocessor.py`: Temporal alignment, regular 15-min resampling, unit conversion to % VWC, train/val/test splitting (70% / 15% / 15%).
  - `src/agri_drift/data/synthetic_generator.py`: Deterministic multi-sensor soil moisture simulator based on physical drying equations (exponential decay), diurnal fluctuations, and rain/irrigation infiltration pulses.
* **Verification Criteria**:
  - Data audit document (`docs/dataset_audit.md`) completed with clear limitations noted.
  - Synthetic dataset generated deterministically with 4 spatially correlated sensor channels over $\ge 60$ days.
  - Chronological splits confirm strict temporal separation with zero future-data leakage.

#### Milestone M2: Labelled Fault-Injection Engine
* **Objective**: Create a deterministic, configurable fault-injection engine for the 5 target fault types with exact ground-truth manifests.
* **Required Inputs**: Preprocessed clean baseline dataset, `configs/fault_injection.yaml`.
* **Source Implementation**:
  - `src/agri_drift/faults/injector.py`: Functions for:
    1. Linear drift ($\pm 0.5\%$ to $\pm 2.0\%$ VWC/day).
    2. Constant bias offset ($\pm 3\%$ to $\pm 10\%$ VWC).
    3. Stuck-at values (freeze at current, zero, or max).
    4. Transient spikes (impulsive spikes with duration 1–3 steps).
    5. Increasing / excessive Gaussian noise.
  - `src/agri_drift/faults/manifest.py`: Serializer for fault ground truth (start/end timestamp, target sensor, fault type, severity).
  - `scripts/generate_fault_benchmarks.py`: CLI script generating standard benchmark suites.
* **Verification Criteria**:
  - Injected faults match exact physical units (explicit % VWC/day).
  - Visualization script produces plots showing clean vs corrupted signals with fault onset annotations.
  - Unit tests verify deterministic generation given random seed.

#### Milestone M3: Statistical Baselines and Sequential Detectors
* **Objective**: Build individual detector components: Range/Rate threshold baseline, OLS Neighbour Residual predictor, and CUSUM / Page–Hinkley change detectors.
* **Required Inputs**: Train and validation data partitions, fault benchmark datasets.
* **Source Implementation**:
  - `src/agri_drift/detectors/threshold.py`: Configurable min/max range and max rate-of-change.
  - `src/agri_drift/detectors/neighbour.py`: Linear regression model trained strictly on healthy training data, predicting target sensor from neighbours, with contamination isolation.
  - `src/agri_drift/detectors/cusum.py`: Two-sided tabular CUSUM with standardized residuals, allowance $k$, threshold $h$, and state reset.
  - `src/agri_drift/detectors/page_hinkley.py`: Streaming Page–Hinkley change detector.
* **Verification Criteria**:
  - Zero false alarms on stable, healthy validation data under standard noise.
  - Correct trigger of CUSUM on positive/negative drift within expected delay.
  - Contamination defense prevents a faulty sensor from triggering alarms on healthy neighbours.
  - Unit tests verify mathematical correctness of CUSUM and Page-Hinkley algorithms.

#### Milestone M4: Physics-Aware Hybrid Decision Engine
* **Objective**: Integrate hydrological physics consistency checking with multi-detector evidence into an interpretable health state machine.
* **Required Inputs**: Event logs (rain/irrigation), detector predictions from M3.
* **Source Implementation**:
  - `src/agri_drift/detectors/physics.py`: Response window analysis post-rain/irrigation; accumulates suspicion when target sensor fails to respond while neighbours respond.
  - `src/agri_drift/detectors/hybrid.py`: Multi-evidence fusion engine outputting:
    - `HEALTHY`
    - `SUSPECTED_FAULT`
    - `CONFIRMED_FAULT`
    - `UNCERTAIN`
    - `OFFLINE_OR_MISSING`
  - Subtype attribution logic (drift, bias, stuck, spike, noise) and plain-English explanation generator.
* **Verification Criteria**:
  - Realistic rain/irrigation events do NOT trigger false alarms in healthy sensors.
  - A non-responsive sensor during watering is correctly flagged as suspicious.
  - Explanations match diagnostic evidence (e.g., "Residual diverged by +4.2% VWC, positive CUSUM exceeded limit").

#### Milestone M5: Research Experiments and Benchmarking
* **Objective**: Conduct systematic comparative experiments across all fault types, evaluate standard metrics, and generate publication-quality figures and tables.
* **Required Inputs**: Test partition, fault benchmark suite, multiple seeds.
* **Source Implementation**:
  - `src/agri_drift/evaluation/metrics.py`: Precision, Recall, F1, Mean Detection Delay (MDD), False Alarms per Sensor-Week (FAR).
  - `src/agri_drift/evaluation/benchmark.py`: Comparative evaluator running:
    1. Fixed Threshold Baseline
    2. Neighbour Residual + CUSUM
    3. Neighbour Residual + Page–Hinkley
    4. Hybrid Physics-Aware Detector
  - `scripts/run_benchmarks.py`: Automated experiment runner generating CSV tables and matplotlib figures (`results/metrics/` and `results/figures/`).
* **Verification Criteria**:
  - Evaluation executed strictly on held-out test data.
  - Negative control experiments included (healthy dynamic drying and rain cycles).
  - Output CSVs and high-resolution PNG/PDF plots generated and saved.

#### Milestone M6: Replay Engine and Interactive Dashboard
* **Objective**: Build a simulated streaming replay engine, SQLite persistence layer, and interactive Streamlit web dashboard.
* **Required Inputs**: Processed test time-series, SQLite database.
* **Source Implementation**:
  - `src/agri_drift/storage/database.py`: SQLite schema and ORM/query helpers for readings, detector states, and alerts.
  - `src/agri_drift/streaming/replay.py`: Chronological tick-by-tick streamer with configurable playback speed and pause/resume.
  - `src/agri_drift/dashboard/app.py`: Streamlit application featuring:
    - Mode badge (`SIMULATION (Replay)` vs `REAL HARDWARE`).
    - Sensor network health overview cards.
    - Synchronized moisture time-series with fault onset and irrigation markers.
    - Live residual and CUSUM tracking plots.
    - Historical alert log with plain-language explanations.
    - Experiment summary view.
* **Verification Criteria**:
  - Replay engine feeds identical detection pipeline as live mode.
  - Dashboard runs locally via `streamlit run src/agri_drift/dashboard/app.py` without errors.
  - SQLite correctly logs all events, states, and alerts.

#### Milestone M7: Raspberry Pi Hardware Integration (Mock + Deployable)
* **Objective**: Implement the hardware adapter layer for Raspberry Pi (ADS1115 ADC, capacitive/resistive probes, DHT22), provide complete mock drivers for laptop validation, and write deployment artifacts.
* **Required Inputs**: Hardware pinout specifications, ADC calibration curves.
* **Source Implementation**:
  - `src/agri_drift/hardware/base.py`: Abstract `SensorReader` base class.
  - `src/agri_drift/hardware/ads1115_reader.py`: CircuitPython ADS1115 driver with graceful fallback.
  - `src/agri_drift/hardware/mock_reader.py`: Mock hardware driver generating voltage/calibrated moisture for testing without physical Pi.
  - `deployment/agri_drift.service`: Systemd service unit definition for continuous edge operation.
  - `deployment/install_pi.sh`: Setup script for Raspberry Pi OS.
  - `docs/hardware_bench_protocol.md`: Step-by-step pot experiment protocol.
* **Verification Criteria**:
  - Mock reader passes integration tests simulating sensor reads, disconnections, and ADC saturation.
  - Hardware modules import safely on laptop without `board` or `adafruit_ads1x15` installed.
  - Clear documentation of real vs simulated boundaries.

#### Milestone M8: Final Documentation and Reproducibility Package
* **Objective**: Consolidate documentation, audit trails, and execution scripts for a fully reproducible research-grade package.
* **Required Inputs**: Completed modules M0–M7.
* **Source Implementation**:
  - Comprehensive `README.md` with quickstart, CLI usage, and reproduction guide.
  - Reproduction script `scripts/reproduce_all.sh` running full pipeline from data generation to benchmark plots.
  - Final experiment results report summarizing findings and answering the central research question.
* **Verification Criteria**:
  - Fresh checkout can reproduce all metrics and figures via a single command.
  - All unit and integration tests pass cleanly.


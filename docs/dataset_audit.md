# Dataset Suitability Audit & Data Specification
## Candidate Datasets vs. Synthetic Multi-Sensor Soil Hydrology

### 1. Evaluation of Candidate Datasets

The project proposal identifies two candidate baseline datasets for healthy agricultural time-series data:
1. **International Soil Moisture Network (ISMN)** (`https://ismn.earth`)
2. **Mendeley Data Low-Cost Soil Moisture Calibration Dataset** (`doi:10.17632/`)

Below is the rigorous evaluation of both candidates against the structural requirements of spatial-neighbour drift detection.

---

### 2. Candidate Evaluation Matrix

| Criterion | Requirement | ISMN Network Data | Mendeley Calibration Data | Synthetic Demonstration Hydrology |
| :--- | :--- | :--- | :--- | :--- |
| **Continuous Time-Series** | Multi-week / multi-month continuous regular sampling (e.g., $\le 1$ hour). | **YES**: Hourly or sub-hourly data spanning years across global networks. | **PARTIAL**: Often discontinuous bench sessions or short calibration runs. | **YES**: Fully continuous, configurable duration (e.g., 90 days at 15-min intervals). |
| **$\ge 3$ Comparable Sensors** | At least 3 horizontally comparable sensors experiencing similar soil moisture. | **NO (Critical Caveat)**: Most stations deploy sensors at *different depths* (5 cm, 10 cm, 20 cm, 50 cm, 100 cm). Sensors at different depths are **not** interchangeable neighbours! | **VARIABLE**: Often tests multiple sensor brands, but not necessarily simultaneous in-pot agricultural replicates. | **YES**: 4 identical simulated probes placed in the same soil horizon / adjacent pots. |
| **Appropriate Timestamps** | Standard ISO-8601 UTC timestamps with monotonic progression. | **YES**: Standardized UTC timestamps with quality flags. | **VARIABLE**: Often local timestamps or elapsed test seconds without calendar reference. | **YES**: Monotonic UTC timestamps on a strict 15-minute grid. |
| **Known Measurement Units** | Explicit physical units (e.g., $m^3/m^3$ or % VWC). | **YES**: Standardized to $m^3/m^3$ (range 0.0 to 0.6). | **VARIABLE**: Often raw ADC counts or voltage with empirical curve fits. | **YES**: Standardized to Volumetric Water Content (% VWC, $0\% - 55\%$). |
| **Rainfall / Irrigation Information** | Logged precipitation or watering events for physics check. | **YES**: Co-located meteorological rain gauges are common in networks (e.g., SCAN, USCRN). | **NO**: Controlled laboratory water additions often lack timestamped event logging. | **YES**: Deterministic rain and irrigation schedule with precise timestamps and applied mm. |
| **Healthy Reference Intervals** | Guaranteed healthy baseline period for training regression models. | **YES**: Historical intervals with ISMN automated quality flags (e.g., 'G' for good). | **UNCERTAIN**: Sensor degradation or calibration drift is often the subject under test. | **YES**: First 60%–70% of time-series is guaranteed pristine healthy baseline. |
| **Accessible without Credential Gate** | Open, automated, reproducible download without manual approval. | **NO**: ISMN requires individual registered user account and email verification for data exports. | **YES**: Openly downloadable via direct DOI URL. | **YES**: Fully self-contained, reproducible, zero external network dependency. |

---

### 3. Critical Hydrological Finding: The Depth-Separation Fallacy

A frequent error in applying multi-sensor drift detection to ISMN datasets is treating sensors installed at different soil depths (e.g., 5 cm vs 20 cm vs 50 cm) as interchangeable spatial neighbours for linear regression.

#### Why Depth Replicating Fails:
1. **Hydraulic Infiltration Lag**: A surface rain event reaches a 5 cm sensor in minutes to hours, but takes 24–72 hours to reach 50 cm.
2. **Evapotranspiration Attenuation**: Diurnal drying cycles caused by solar radiation and plant transpiration are pronounced in the top 10 cm but virtually undetectable at 50 cm.
3. **Soil Horizon Differences**: Texture, bulk density, and organic matter vary dramatically across soil horizons, creating non-linear, hysteretic relationships rather than linear spatial correlation.

> [!IMPORTANT]
> To test spatial neighbour residual detection honestly, sensors must be **horizontal peers** (same depth, same soil type, within a short spatial correlation radius of 0.2 m to 10 m, or in adjacent pots receiving identical watering).
>
> If ISMN data is used, it can only supply the *environmental forcing series* (rainfall, ambient temperature, drying trends) or require non-linear vertical infiltration modeling. Consequently, for direct multi-sensor edge diagnosis, a **physically derived, multi-probe demonstration dataset** is required to guarantee horizontal comparability without fabricating real-world claims.

---

### 4. Synthetic Multi-Sensor Soil Hydrology Specification

To provide a rigorous, transparent, and reproducible testbed, we implement a physical soil hydrology simulation based on the 1-D soil water balance equation:

$$\frac{d\theta_i(t)}{dt} = - ET(t) - K(\theta_i) + I_i(t) + R(t)$$

Where:
- $\theta_i(t)$ is the Volumetric Water Content (% VWC) of probe $i \in \{1, 2, 3, 4\}$.
- $ET(t) = ET_{max} \cdot \max(0, \sin(\frac{2\pi (t - t_{sunrise})}{24})) \cdot \frac{\theta_i(t) - \theta_{pwp}}{\theta_{fc} - \theta_{pwp}}$ simulates diurnal solar evapotranspiration, bounded by field capacity ($\theta_{fc} \approx 36\%$) and permanent wilting point ($\theta_{pwp} \approx 10\%$).
- $K(\theta_i)$ represents unsaturated gravity drainage when $\theta_i > \theta_{fc}$.
- $I_i(t)$ is controlled irrigation pulses (applied at scheduled intervals).
- $R(t)$ is episodic stochastic rainfall events.

#### Spatial Correlation Structure:
- Common environmental signal $\theta_{true}(t)$ shared by all probes in the same root zone.
- Individual probe installation offsets $\delta_i \sim \mathcal{N}(0, 0.8\%)$ accounting for minor local soil compaction variance.
- Common spatial fluctuation correlation $\rho \ge 0.88$.
- Independent zero-mean Gaussian electronic noise $\epsilon_{i,t} \sim \mathcal{N}(0, 0.25\%)$.

---

### 5. Unified Data Schema

All data loaders (CSV, Parquet, Synthetic) normalize outputs to this canonical schema:

| Column Name | Type | Physical Unit | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | `datetime64[ns, UTC]` | ISO-8601 UTC | Continuous monotonic timestamp at 15-minute sampling interval |
| `sensor_1` | `float64` | % VWC ($0.0 - 100.0$) | Probe 1 soil volumetric water content |
| `sensor_2` | `float64` | % VWC ($0.0 - 100.0$) | Probe 2 soil volumetric water content |
| `sensor_3` | `float64` | % VWC ($0.0 - 100.0$) | Probe 3 soil volumetric water content |
| `sensor_4` | `float64` | % VWC ($0.0 - 100.0$) | Probe 4 soil volumetric water content |
| `rain_mm` | `float64` | Millimeters ($mm$) | Accumulated rainfall over the 15-min interval |
| `irrigation_mm`| `float64` | Millimeters ($mm$) | Applied irrigation water over the interval |
| `ambient_temp` | `float64` | Degrees Celsius ($^\circ C$) | Ambient air temperature |
| `relative_humidity`| `float64` | Percent ($0.0 - 100.0$) | Ambient relative humidity |

---

### 6. Partitioning Protocol (No Future Data Leakage)

Data is chronologically partitioned into three non-overlapping contiguous segments:
1. **Train Set (First 70%)**: Strictly pristine, healthy readings. Used to fit OLS neighbour regression models, establish baseline residual variance $\sigma_0^2$, and calibrate CUSUM parameters.
2. **Validation Set (Next 15%)**: Uncorrupted healthy data containing natural rain/irrigation cycles. Used to verify that CUSUM/Page-Hinkley and physics checks generate $\le 0.05$ false alarms per sensor-week (negative-control tuning).
3. **Test Set (Final 15%)**: Evaluates detection performance under controlled single-fault and multi-fault injection scenarios. Absolutely no parameters, thresholds, or regression coefficients may be updated using the test partition.


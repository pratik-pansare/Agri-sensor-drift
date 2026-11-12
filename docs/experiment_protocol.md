# Experimental Evaluation Protocol & Benchmark Methodology
## Sensor-Fault and Drift Detection in Agricultural IoT

### 1. Research Objectives & Formal Hypotheses

The primary research objective is to determine whether an edge node can differentiate genuine environmental dynamics (soil drying, heavy rain, drip irrigation) from in-situ sensor faults (gradual electrochemical drift, bias, stuck-at, spikes, noise) by fusing:
1. Spatial cross-prediction from neighbouring sensors.
2. Sequential change-point statistics (CUSUM, Page–Hinkley).
3. Hydrological physical response expectations.

#### Formal Hypotheses:
* **$H_1$ (Drift Detection Superiority)**: Neighbour-residual CUSUM will achieve $\ge 85\%$ F1-score and significantly shorter detection delay on gradual drift compared to fixed threshold baselines (which remain blind while readings stay in the 10%–45% VWC range).
* **$H_2$ (Negative-Control Robustness)**: By cross-referencing spatial consensus and known rainfall/irrigation events, the hybrid decision engine maintains a false alarm rate $\le 0.10$ false alarms per sensor-week during natural environmental shocks (rain surges and rapid drying).
* **$H_3$ (Contamination Resistance)**: When one sensor experiences severe failure, the dynamic masking in the neighbour predictor prevents false alarms from propagating to healthy neighbouring sensors.

---

### 2. Evaluated Systems & Baselines

| Model Identifier | Description | Key Hyperparameters |
| :--- | :--- | :--- |
| **`Baseline-Threshold`** | Traditional static range and rate-of-change checker. | Range: $[5.0\%, 50.0\%]$ VWC; Max rate: $\Delta \le 8.0\%$ VWC / 15-min. |
| **`Neighbour-CUSUM`** | Linear regression on healthy neighbours; two-sided tabular CUSUM on standardized residuals. | Regression: OLS; CUSUM allowance $k = 0.5\sigma_0$, threshold $h = 5.0\sigma_0$. |
| **`Neighbour-PageHinkley`** | Linear regression on healthy neighbours; Page–Hinkley test on residual stream. | Regression: OLS; PH $\delta = 0.5\sigma_0$, threshold $\lambda = 15.0\sigma_0$. |
| **`Hybrid-Physics-Aware`** | Multi-evidence fusion: Thresholds + Neighbour CUSUM + Page-Hinkley + Rain/Irrigation response check + Persistence filter. | Confirmation window: 3 consecutive steps; Physics response window: 4 hours, min delta: $1.5\%$ VWC. |

---

### 3. Evaluation Metrics

#### 3.1 Sample-Level and Event-Level Classification
* **True Positive ($TP$)**: Fault state correctly flagged during ground-truth fault period.
* **False Positive ($FP$)**: Fault flagged during healthy period.
* **False Negative ($FN$)**: Sensor operating under fault but flagged as HEALTHY.
* **True Negative ($TN$)**: Healthy reading correctly identified as HEALTHY.
* **Precision**: $P = \frac{TP}{TP + FP}$
* **Recall**: $R = \frac{TP}{TP + FN}$
* **$F_1$-Score**: $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$

#### 3.2 Temporal and Operational Edge Metrics
* **Mean Detection Delay (MDD)**: The time elapsed (in hours) between the injection onset timestamp $t_{start}$ and the first alert timestamp $t_{alert}$:
  $$MDD = \frac{1}{N_{detected}} \sum_{k=1}^{N_{detected}} (t_{alert, k} - t_{start, k})$$
* **False Alarm Rate (FAR)**: Number of false alarm events per sensor-week on healthy validation/test intervals:
  $$FAR = \frac{\text{Total False Alarm Events}}{\text{Total Sensor-Days Elapsed} / 7}$$
* **Computational Footprint**: Peak memory (MB) and mean execution latency per 15-min sample step ($\mu s$) on a standard CPU core.

---

### 4. Fault Parameter Sweep Grid

Faults are injected across a standardized grid of severities and durations:

| Fault Type | Parameter | Low Severity | Medium Severity | High Severity |
| :--- | :--- | :--- | :--- | :--- |
| **Gradual Drift** | Slope ($m$) in % VWC / day | $+0.5\%$ / day | $+1.5\%$ / day | $+3.0\%$ / day (and negative equivalents) |
| **Constant Bias** | Additive offset ($b$) | $+3.0\%$ VWC | $+6.0\%$ VWC | $+12.0\%$ VWC |
| **Stuck-at** | Clamped value | Freeze at current | Ground / $0.0\%$ | Saturation / $55.0\%$ |
| **Transient Spikes**| Impulse amplitude ($\Delta$) | $\pm 8.0\%$ VWC | $\pm 15.0\%$ VWC | $\pm 30.0\%$ VWC (1–2 samples) |
| **Excessive Noise** | Standard deviation ($\sigma$) | $2 \times \sigma_0$ | $4 \times \sigma_0$ | $8 \times \sigma_0$ |

---

### 5. Negative-Control Experiment Protocol

To guarantee that detectors do not trigger false alarms during natural environmental fluctuations, the test suite includes explicit negative controls:
1. **Sudden Rain Deluge**: Simultaneous surge across all sensors ($+15\%$ to $+25\%$ VWC within 30 minutes). Because all sensors experience the surge simultaneously, spatial residuals remain centered near zero, and no fault must be raised.
2. **Post-Rain Soil Drainage**: Fast gravity drainage followed by slow evapotranspiration. Tests that diurnal oscillations do not trigger drift detectors.
3. **Controlled Drip Irrigation**: Periodic pulsed water application in specific soil zones, verified against logged irrigation records.

---

### 6. Reproducibility & Output Artifacts

* **Multi-Seed Testing**: Every experiment is evaluated across 5 random seeds (`[42, 101, 777, 1337, 2026]`) to compute mean and standard error.
* **Tabular Results**: Saved as machine-readable CSV files in `results/metrics/`:
  - `overall_comparison.csv`
  - `metrics_by_fault_type.csv`
  - `detection_delay_summary.csv`
  - `ablation_physics_check.csv`
* **Publication Figures**: Rendered in both 300 DPI PNG and vector PDF format in `results/figures/`:
  - `figure1_fault_signatures.png` / `.pdf`: Clean vs corrupted time-series for all 5 fault types.
  - `figure2_pr_curves.png` / `.pdf`: Precision-Recall curves by model and fault type.
  - `figure3_detection_delay_boxplots.png` / `.pdf`: Detection delay across drift rates.
  - `figure4_live_diagnostics.png` / `.pdf`: Synchronized time-series, neighbour residuals, CUSUM statistics, and health state flags.


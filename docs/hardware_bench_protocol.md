# Bench-Testbed Hardware Protocol & Pot Experiment
## Physical Validation Guide for Raspberry Pi & Multi-Sensor Array

### 1. Hardware Architecture & Electrical Pinout

```
+-----------------------------------------------------------+
|                     Raspberry Pi 4 / 5                    |
|   Pin 1 (3.3V Power)  -----> ADS1115 VDD                  |
|   Pin 3 (I2C SDA)     -----> ADS1115 SDA                  |
|   Pin 5 (I2C SCL)     -----> ADS1115 SCL                  |
|   Pin 9 (Ground)      -----> ADS1115 GND                  |
|   Pin 2 (5.0V Power)  -----> DHT22 VCC (Optional)         |
|   Pin 7 (GPIO 4)      -----> DHT22 Data (Optional)        |
+-----------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------+
|                     ADS1115 16-bit ADC                    |
|   A0 (Channel 0) <----- Capacitive Probe 1 (Healthy)      |
|   A1 (Channel 1) <----- Capacitive Probe 2 (Healthy)      |
|   A2 (Channel 2) <----- Capacitive Probe 3 (Healthy)      |
|   A3 (Channel 3) <----- Resistive Probe 4 (Degraded)      |
+-----------------------------------------------------------+
```

---

### 2. Experimental Setup Protocol

#### 2.1 Soil & Pot Preparation
1. **Soil Homogenization**: Use uniform sandy-loam potting soil, mixed thoroughly and sieved through a 2 mm mesh to eliminate large aggregates or air voids.
2. **Pots**: Fill four identical 2-liter pots with equal dry soil weight (e.g., exactly $1.50\text{ kg} \pm 0.01\text{ kg}$).
3. **Sensor Placement**:
   - Install **Probe 1** (Capacitive v1.2) vertically in Pot 1 at 5 cm depth.
   - Install **Probe 2** (Capacitive v1.2) vertically in Pot 2 at 5 cm depth.
   - Install **Probe 3** (Capacitive v1.2) vertically in Pot 3 at 5 cm depth.
   - Install **Probe 4** (Resistive Probe with exposed copper traces) in Pot 4 at 5 cm depth.
4. **Deliberate Degradation**:
   - Probe 4 is deliberately subjected to continuous DC electrolysis or mechanical coating abrasion to induce accelerated electrochemical corrosion and rapid positive/negative baseline drift over 3–7 days.

---

### 3. Sensor Calibration Procedure (Two-Point Gravimetric)

To convert raw ADC voltage $V_{raw}$ to percentage Volumetric Water Content (% VWC):

1. **Air Calibration ($V_{dry}$)**:
   - Hold clean, dry probe in air. Record raw ADC voltage $V_{dry}$ (typically $\approx 2.85\text{ V}$).
2. **Water Saturation Calibration ($V_{wet}$)**:
   - Submerge probe sensing blades in room-temperature water up to the white guideline. Record $V_{wet}$ (typically $\approx 1.25\text{ V}$).
3. **Transfer Function**:
   $$\theta(\%) = \min\left(55.0, \max\left(0.0, \frac{V_{dry} - V_{raw}}{V_{dry} - V_{wet}} \times 55.0\right)\right)$$
4. **Gravimetric Ground-Truth Reference**:
   - Weigh each pot on a digital kitchen scale ($\pm 1\text{ g}$) daily.
   - Gravimetric Water Content ($GWC$):
     $$GWC = \frac{W_{current} - W_{dry\_soil}}{W_{dry\_soil}} \times 100\%$$
   - Multiply by bulk density ($\rho_b \approx 1.2\text{ g/cm}^3$) to yield true reference $VWC$.

---

### 4. Controlled Watering Regime (7-Day Bench Protocol)

| Day | Time (UTC) | Action | Target Expected Moisture | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Day 1** | 09:00 | Add 150 mL water to all 4 pots | $+8\%$ to $+10\%$ VWC surge | Baseline establishment; all sensors should respond together. |
| **Day 2** | — | Natural drying | $-2\%$ to $-3\%$ VWC / day | Evaporative recession. |
| **Day 3** | 09:00 | Add 200 mL water to Pots 1, 2, 3 only | Pots 1–3 rise $+12\%$; Pot 4 flat | Tests spatial neighbour divergence and physics check. |
| **Day 4** | — | Deliberate corrosion trigger on Probe 4 | Probe 4 exhibits $+1.5\%$ VWC/day drift | CUSUM / Page-Hinkley should trigger on Probe 4. |
| **Day 5** | 14:00 | Simulated rain surge (250 mL all pots) | High surge across all pots | Negative control test: consensus surge must NOT raise false alarm. |
| **Day 6** | — | Disconnect A3 wire momentarily | Reading drops to NaN / $0.0\text{ V}$ | Tests stuck-at / OFFLINE_OR_MISSING transition. |
| **Day 7** | — | End of bench trial | Soil drying | Compute bench test metrics against gravimetric scale logs. |


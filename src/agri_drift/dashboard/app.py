"""Streamlit Web Dashboard for Sensor-Fault and Drift Detection in Agricultural IoT."""

import os
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np

from agri_drift.storage.database import DatabaseManager
from agri_drift.data.synthetic_generator import SyntheticHydrologyGenerator
from agri_drift.faults.injector import FaultInjector
from agri_drift.detectors.threshold import ThresholdDetector
from agri_drift.detectors.neighbour import NeighbourResidualDetector
from agri_drift.detectors.physics import PhysicsConsistencyDetector
from agri_drift.detectors.hybrid import HybridDecisionEngine
from agri_drift.streaming.replay import ReplayStreamer

st.set_page_config(
    page_title="Agri-Drift IoT Diagnostics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = Path("data/agri_drift.db")
db = DatabaseManager(db_path=DB_PATH)

# Initialize Session State
if "streamer" not in st.session_state:
    # Set up demonstration dataset and engine
    gen = SyntheticHydrologyGenerator(days=30, interval_minutes=15, random_seed=42)
    clean_df, events = gen.generate()

    # Inject demo faults
    injector = FaultInjector(random_seed=42)
    demo_scenarios = [
        {"id": "demo_drift", "fault_type": "drift", "sensor_id": "sensor_2", "rate_vwc_per_day": 1.5, "start_hour": 72, "duration_hours": 96},
        {"id": "demo_bias", "fault_type": "bias", "sensor_id": "sensor_1", "offset_vwc": 5.0, "start_hour": 200, "duration_hours": 48},
        {"id": "demo_stuck", "fault_type": "stuck_at", "sensor_id": "sensor_4", "mode": "freeze_current", "start_hour": 280, "duration_hours": 36},
    ]
    corrupted_df, _ = injector.inject_benchmark_suite(clean_df, demo_scenarios)

    sensors = ["sensor_1", "sensor_2", "sensor_3", "sensor_4"]
    neighbour_det = NeighbourResidualDetector(target_sensors=sensors)
    neighbour_det.fit(clean_df.iloc[:2000])

    engine = HybridDecisionEngine(
        target_sensors=sensors,
        threshold_detector=ThresholdDetector(min_vwc=3.0, max_vwc=55.0, max_rate=8.0),
        neighbour_detector=neighbour_det,
        confirmation_window_steps=3,
    )

    streamer = ReplayStreamer(df=corrupted_df, engine=engine, db_manager=db)
    # Prime database with first 120 steps (5 days)
    for _ in range(120):
        streamer.step()

    st.session_state.streamer = streamer
    st.session_state.events = events

streamer: ReplayStreamer = st.session_state.streamer

# Header & Mode Banner
st.title("🌱 Sensor-Fault & Drift Detection in Agricultural IoT")
mode = "SIMULATION (Deterministic Replay)"
st.info(f"**OPERATING MODE: {mode}** — Replaying multi-probe soil moisture time-series with live diagnostic fusion.")

# Sidebar Controls
st.sidebar.header("🕹️ Replay Streamer Controls")
st.sidebar.markdown(f"**Replay Progress**: Step `{streamer.current_index}` of `{streamer.total_steps}`")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("⏩ Step +1 (15m)", use_container_width=True):
        streamer.step()
        st.rerun()

with col_sb2:
    if st.button("⏩ Step +12 (3h)", use_container_width=True):
        for _ in range(12):
            streamer.step()
        st.rerun()

if st.sidebar.button("🔄 Reset Replay", use_container_width=True):
    streamer.reset()
    for _ in range(50):
        streamer.step()
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("### 📊 Health State Legend")
st.sidebar.markdown("🟢 **HEALTHY**: Consistent with neighbours & physics")
st.sidebar.markdown("🟡 **SUSPECTED_FAULT**: Residual or CUSUM shift active")
st.sidebar.markdown("🔴 **CONFIRMED_FAULT**: Persistent divergence confirmed")
st.sidebar.markdown("⚪ **OFFLINE_OR_MISSING**: Signal loss or disconnect")

# 1. Sensor Overview Cards
st.subheader("📡 Real-Time Sensor Network Status")
readings_df = db.get_recent_readings(limit_per_sensor=10)

cols = st.columns(4)
sensors = ["sensor_1", "sensor_2", "sensor_3", "sensor_4"]

state_colors = {
    "HEALTHY": "🟢",
    "SUSPECTED_FAULT": "🟡",
    "CONFIRMED_FAULT": "🔴",
    "UNCERTAIN": "🟠",
    "OFFLINE_OR_MISSING": "⚪",
}

for i, s in enumerate(sensors):
    with cols[i]:
        curr_state = streamer.engine.current_states.get(s).value
        icon = state_colors.get(curr_state, "⚪")

        s_readings = readings_df[readings_df["sensor_id"] == s]
        val = s_readings.iloc[-1]["vwc_percent"] if not s_readings.empty else 0.0
        prev_val = s_readings.iloc[-2]["vwc_percent"] if len(s_readings) >= 2 else val
        delta = round(val - prev_val, 2)

        st.metric(
            label=f"{icon} {s.upper()} ({curr_state})",
            value=f"{val:.2f}% VWC",
            delta=f"{delta:+.2f}%",
        )

# 2. Synchronized Moisture Trend Plots
st.subheader("📈 Multi-Sensor Soil Moisture Dynamics")
history_df = db.get_recent_readings(limit_per_sensor=150)

if not history_df.empty:
    pivot_df = history_df.pivot(index="timestamp", columns="sensor_id", values="vwc_percent")
    st.line_chart(pivot_df, use_container_width=True)

# 3. Deep Diagnostics: Residuals & CUSUM Statistics
st.subheader("🔍 Deep Diagnostic Statistics")
selected_sensor = st.selectbox("Select Sensor for Residual & CUSUM Traces", sensors, index=1)
sensor_history = db.get_detector_history(selected_sensor, limit=120)

if not sensor_history.empty:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(f"**Spatial Residual (Actual - Predicted by Neighbours) for {selected_sensor}**")
        st.line_chart(sensor_history.set_index("timestamp")[["residual"]], use_container_width=True)

    with col_d2:
        st.markdown(f"**Sequential CUSUM Traces ($S_t^+, S_t^-$) for {selected_sensor}**")
        st.line_chart(sensor_history.set_index("timestamp")[["cusum_pos", "cusum_neg"]], use_container_width=True)

# 4. Recent Diagnostic Alerts Table
st.subheader("🚨 Farmer-Facing Diagnostic Alerts Log")
alerts_df = db.get_recent_alerts(limit=20)
if not alerts_df.empty:
    st.dataframe(alerts_df, use_container_width=True, hide_index=True)
else:
    st.success("No alerts triggered. All sensors operating within normal parameters.")

# 5. Benchmark Performance View
st.subheader("📑 Empirical Benchmark Summary (M5 Results)")
metrics_path = Path("results/metrics/overall_comparison.csv")
if metrics_path.exists():
    bm_df = pd.read_csv(metrics_path)
    st.dataframe(bm_df, use_container_width=True, hide_index=True)
else:
    st.info("Run `python scripts/run_benchmarks.py` to populate reproducible benchmark metrics.")


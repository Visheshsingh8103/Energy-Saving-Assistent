"""
app.py
WattWise AI: Intelligent Electricity Management System
Tagline: "Detect Waste. Predict Demand. Save Energy."
Full-stack production Streamlit application.
"""

import os
import sys
import time
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import (
    DEFAULT_DB_PATH,
    init_db,
    get_locations,
    get_readings,
    get_all_settings,
    set_setting,
    get_database_summary,
    insert_single_reading,
    log_simulation_event,
    get_simulation_events,
    reset_demo_data,
    save_recommendations,
    get_recommendations,
    save_detective_finding,
    save_detective_findings_bulk,
    get_detective_findings,
    update_investigation_status
)
from src.waste_detective import (
    WasteDetective,
    calculate_waste_metrics,
    compute_evidence_score,
    export_detective_findings_csv
)
from src.data_generator import generate_synthetic_energy_dataset, LOCATION_PROFILES
from src.forecasting import ElectricityForecaster
from src.anomaly_detection import EnergyAnomalyDetector
from src.waste_intelligence import EnergyWasteIntelligenceEngine
from src.savings_calculator import simulate_room_savings, get_preset_scenario
from src.bill_estimator import calculate_electricity_bill, estimate_bill_from_dataframe
from src.recommendations import RecommendationEngine
from src.reporting import (
    export_consumption_csv,
    export_anomalies_csv,
    export_recommendations_csv,
    generate_executive_summary_markdown
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="WattWise AI — Energy Management",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Premium Custom Styling
st.markdown("""
<style>
    /* Global Container Aesthetics */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #38bdf8;
        margin-top: 0.25rem;
        font-weight: 500;
    }
    .demo-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-left: 0.75rem;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #334155;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1.2;
    }
    .metric-subtext {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.35rem;
    }
    .status-badge-ok {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-badge-warn {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-badge-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .alert-banner {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid #ef4444;
        border-left: 6px solid #ef4444;
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .success-banner {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid #10b981;
        border-left: 6px solid #10b981;
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Database & State
@st.cache_resource
def setup_database_environment():
    """Ensure database schema is created and initialized."""
    init_db(DEFAULT_DB_PATH)
    return True

setup_database_environment()

# Initialize session state variables
if "demo_playing" not in st.session_state:
    st.session_state.demo_playing = False
if "demo_step" not in st.session_state:
    st.session_state.demo_step = 0
if "last_injected_alert" not in st.session_state:
    st.session_state.last_injected_alert = None
if "active_location" not in st.session_state:
    st.session_state.active_location = "computer_lab"


# Data loading helpers
@st.cache_data(ttl=60)
def load_locations_data():
    return get_locations(DEFAULT_DB_PATH)

@st.cache_data(ttl=60)
def load_all_readings(location_id=None, start_date=None, end_date=None):
    return get_readings(DEFAULT_DB_PATH, location_id=location_id, start_date=start_date, end_date=end_date)

@st.cache_data(ttl=60)
def load_current_settings():
    return get_all_settings(DEFAULT_DB_PATH)


locations = load_locations_data()
settings = load_current_settings()
loc_map = {loc["id"]: loc["name"] for loc in locations}
loc_meta_map = {loc["id"]: loc for loc in locations}


# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=300&q=80", use_container_width=True)
    st.markdown("### ⚡ **WattWise AI Navigation**")
    
    nav_option = st.radio(
        "Select Application Section:",
        [
            "📊 Executive Overview",
            "🚨 Live Demo Monitor",
            "🧠 Waste Intelligence Engine",
            "🕵️ Energy Waste Detective",
            "📈 AI Forecasting",
            "🔬 Energy Savings Lab",
            "🏢 Room Comparison",
            "💵 Electricity Bill Estimator",
            "📑 Reports & Settings"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### ⚙️ **Active Configuration**")

    # Global Location Filter
    loc_options = ["all"] + [l["id"] for l in locations]
    loc_labels = ["All Locations (Aggregate)"] + [l["name"] for l in locations]
    selected_loc_idx = st.selectbox(
        "Monitored Facility:",
        range(len(loc_options)),
        format_func=lambda i: loc_labels[i],
        index=1 if st.session_state.active_location in loc_options else 0
    )
    current_loc_filter = loc_options[selected_loc_idx]

    # Global Tariff Rate
    active_tariff = st.number_input(
        f"Tariff Rate ({settings.get('currency_symbol', '$')}/kWh):",
        min_value=0.01,
        max_value=2.00,
        value=float(settings.get("tariff_per_kwh", 0.15)),
        step=0.01,
        help="Configurable utility electricity price per kilowatt-hour."
    )

    # Anomaly Sensitivity Slider
    anomaly_sensitivity = st.slider(
        "Anomaly Sensitivity Rate:",
        min_value=0.01,
        max_value=0.15,
        value=float(settings.get("anomaly_sensitivity", 0.05)),
        step=0.01,
        help="IsolationForest contamination parameter (higher value detects more subtle deviations)."
    )

    st.markdown("---")
    st.markdown("#### 🛡️ **System Telemetry Health**")
    db_summary = get_database_summary(DEFAULT_DB_PATH)
    st.markdown(f"""
    - **Database:** `SQLite Active`
    - **Telemetry Records:** `{db_summary.get('readings_count', 0):,}`
    - **Monitored Rooms:** `{db_summary.get('locations_count', 0)}`
    - **Status:** <span class="status-badge-ok">OPERATIONAL</span>
    """, unsafe_allow_html=True)


# Application Header Banner
st.markdown(f"""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 class="main-title">⚡ WATTWISE AI</h1>
            <div class="main-subtitle">Detect Waste. Predict Demand. Save Energy.</div>
        </div>
        <div>
            <span class="demo-pill">🟢 DEMO MODE (SYNTHETIC TELEMETRY)</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: EXECUTIVE OVERVIEW
# ==============================================================================
if nav_option == "📊 Executive Overview":
    st.subheader("📊 Executive Overview & Energy Telemetry")
    st.caption("Real-time summary of energy consumption, expenditure, and detected anomalies across all monitored educational & laboratory facilities.")

    # Fetch data
    df_readings = load_all_readings(location_id=current_loc_filter)

    if df_readings.empty:
        st.warning("No telemetry records found. Run the data generator script or check database settings.")
    else:
        total_kwh = float(df_readings["consumption_kwh"].sum())
        total_cost = total_kwh * active_tariff
        anomalies_count = int(df_readings["is_anomaly_flag"].sum())
        peak_kwh = float(df_readings["consumption_kwh"].max())
        latest_reading = df_readings.iloc[-1]

        # Top Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Consumed</div>
                <div class="metric-value">{total_kwh:,.1f} <span style="font-size: 1rem; color: #94a3b8;">kWh</span></div>
                <div class="metric-subtext">Across {len(df_readings):,} recorded hours</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Utility Cost</div>
                <div class="metric-value">${total_cost:,.2f}</div>
                <div class="metric-subtext">@ ${active_tariff:.2f} per kWh</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Peak Power Demand</div>
                <div class="metric-value">{peak_kwh:.2f} <span style="font-size: 1rem; color: #94a3b8;">kW</span></div>
                <div class="metric-subtext">Maximum instantaneous load</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Detected Anomalies</div>
                <div class="metric-value">{anomalies_count} <span style="font-size: 1rem; color: #f87171;">flags</span></div>
                <div class="metric-subtext">IsolationForest flagged events</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            # Quick estimate of waste cost
            est_waste_cost = (anomalies_count * 2.5) * active_tariff
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Waste Avoidance</div>
                <div class="metric-value" style="color: #34d399;">${est_waste_cost:,.2f}</div>
                <div class="metric-subtext">Estimated recoverable leakage</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Room Status Cards Grid
        st.markdown("#### 🏢 Monitored Location Status")
        room_cols = st.columns(len(locations))
        for idx, loc in enumerate(locations):
            loc_df = df_readings[df_readings["location_id"] == loc["id"]]
            curr_draw = float(loc_df["consumption_kwh"].iloc[-1]) if not loc_df.empty else 0.0
            loc_anom = int(loc_df["is_anomaly_flag"].sum()) if not loc_df.empty else 0

            with room_cols[idx]:
                badge_class = "status-badge-ok" if loc_anom < 10 else ("status-badge-warn" if loc_anom < 25 else "status-badge-danger")
                status_text = "OPTIMAL" if loc_anom < 10 else ("ELEVATED" if loc_anom < 25 else "AUDIT REQUIRED")
                st.markdown(f"""
                <div class="metric-card" style="padding: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-weight: 700; font-size: 0.95rem; color: #f8fafc;">{loc['name']}</span>
                        <span class="{badge_class}">{status_text}</span>
                    </div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #38bdf8;">{curr_draw:.2f} <span style="font-size: 0.8rem; color: #94a3b8;">kWh/h</span></div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">Type: {loc['type'].replace('_', ' ').title()}</div>
                    <div style="font-size: 0.75rem; color: #64748b;">Anomalies: {loc_anom} events</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Telemetry Time Series Chart
        st.markdown("#### 📈 Historical Energy Consumption Profile (Last 14 Days)")
        
        # Take last 14 days for fast, crisp rendering
        df_chart = df_readings.iloc[-336:].copy()
        
        fig = go.Figure()
        if current_loc_filter == "all":
            for loc in locations:
                sub = df_chart[df_chart["location_id"] == loc["id"]]
                fig.add_trace(go.Scatter(
                    x=sub["timestamp"],
                    y=sub["consumption_kwh"],
                    name=loc["name"],
                    mode="lines",
                    line=dict(width=2)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df_chart["timestamp"],
                y=df_chart["consumption_kwh"],
                name="Hourly Consumption (kWh)",
                mode="lines",
                line=dict(color="#10b981", width=2.5)
            ))
            # Overlay anomalies
            anoms = df_chart[df_chart["is_anomaly_flag"] == 1]
            if not anoms.empty:
                fig.add_trace(go.Scatter(
                    x=anoms["timestamp"],
                    y=anoms["consumption_kwh"],
                    mode="markers",
                    name="Flagged Anomalies",
                    marker=dict(color="#ef4444", size=8, symbol="diamond")
                ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0)",
            plot_bgcolor="rgba(30, 41, 59, 0.7)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Timestamp (Hourly Intervals)", gridcolor="#334155"),
            yaxis=dict(title="Electricity Draw (kWh)", gridcolor="#334155"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# SECTION 2: LIVE DEMO MONITOR
# ==============================================================================
elif nav_option == "🚨 Live Demo Monitor":
    st.subheader("🚨 Live Demonstration Monitor & Interactive Telemetry")
    st.markdown("""
    **Presentation Demonstration Mode:** Simulate live energy telemetry streaming from laboratory and classroom sub-meters.
    Use the controls below to step through continuous readings or **inject a live anomaly** to observe the real-time detection pipeline.
    """)

    col_ctrl, col_display = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### 🎛️ Simulation Controls")
        demo_room = st.selectbox(
            "Select Room for Live Feed:",
            [l["id"] for l in locations],
            format_func=lambda x: loc_map[x],
            index=0
        )

        st.markdown("##### ⚡ Injected Anomaly Test Scenario")
        st.info("Click **'Trigger Demo Anomaly'** to inject a high off-hours power spike into the live telemetry stream and observe how the ML detector and Waste Intelligence Engine respond.")

        if st.button("🚨 Trigger Demo Anomaly", type="primary", use_container_width=True):
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Create a realistic spike: e.g. 5.6 kWh draw at night with 0 occupants
            injected_kwh = 5.65
            injected_reading = {
                "timestamp": now_str,
                "location_id": demo_room,
                "hour": 21, # 9:00 PM
                "day_of_week": 3,
                "month": datetime.now().month,
                "temperature": 27.5,
                "occupancy": 0,
                "consumption_kwh": injected_kwh,
                "is_anomaly_flag": 1,
                "anomaly_score": 0.94,
                "is_demo_injected": 1
            }

            # Insert into database
            insert_single_reading(DEFAULT_DB_PATH, injected_reading)

            # Audit log
            log_simulation_event(
                DEFAULT_DB_PATH,
                location_id=demo_room,
                event_type="DEMO_ANOMALY_TRIGGERED",
                consumption_kwh=injected_kwh,
                details=f"Demo anomaly injected: {injected_kwh} kWh at 21:00 with zero occupancy."
            )

            # Evaluate through Waste Intelligence Engine
            loc_info = loc_meta_map.get(demo_room, {})
            engine = EnergyWasteIntelligenceEngine(
                operating_start=settings.get("operating_hours_start", 8),
                operating_end=settings.get("operating_hours_end", 18),
                tariff_per_kwh=active_tariff
            )

            diag = engine.analyze_event(
                location_id=demo_room,
                location_name=loc_map[demo_room],
                timestamp=now_str,
                consumption_kwh=injected_kwh,
                expected_kwh=loc_info.get("baseline_kw_closed", 0.8),
                occupancy=0,
                temperature=27.5,
                is_anomaly=True,
                anomaly_score=0.94,
                baseline_closed_kwh=loc_info.get("baseline_kw_closed", 0.8),
                baseline_open_kwh=loc_info.get("baseline_kw_open", 4.0)
            )

            st.session_state.last_injected_alert = diag
            st.toast("🚨 Anomaly Injected Successfully!", icon="⚠️")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset Demo State", use_container_width=True):
            reset_demo_data(DEFAULT_DB_PATH)
            st.session_state.last_injected_alert = None
            st.success("Demo events reset. Baseline state restored.")
            time.sleep(1)
            st.rerun()

    with col_display:
        st.markdown("#### 📡 Real-Time Telemetry Stream")

        # Check if an alert was triggered
        alert = st.session_state.last_injected_alert
        if alert:
            st.markdown(f"""
            <div class="alert-banner">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: #ef4444; font-size: 1.15rem;">🚨 {alert['status']}</strong>
                    <span class="status-badge-danger">SEVERITY: {alert['severity']}</span>
                </div>
                <div style="margin-top: 0.5rem; color: #cbd5e1; font-size: 0.95rem;">
                    Location: <strong>{alert['location_name']}</strong> | Measured: <strong>{alert['measured_kwh']:.2f} kWh</strong> (Baseline: {alert['expected_kwh']:.2f} kWh)
                </div>
                <div style="margin-top: 0.25rem; color: #f87171; font-weight: 600;">
                    Unnecessary Excess: +{alert['diff_kwh']:.2f} kWh (Cost Leakage: +${alert['estimated_extra_cost']:.2f})
                </div>
                <div style="margin-top: 0.75rem; color: #e2e8f0; font-size: 0.88rem;">
                    <strong>Root Cause Diagnosis:</strong>
                    <ul style="margin: 0.25rem 0 0.5rem 1.25rem;">
                        {''.join([f"<li>{exp}</li>" for exp in alert['possible_explanations']])}
                    </ul>
                    <strong>Recommended Next Action:</strong>
                    <ul style="margin: 0.25rem 0 0 1.25rem; color: #38bdf8;">
                        {''.join([f"<li>{act}</li>" for act in alert['recommended_actions']])}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-banner">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: #10b981; font-size: 1.1rem;">🟢 Telemetry Stream Normal</strong>
                    <span class="status-badge-ok">ACTIVE MONITORING</span>
                </div>
                <div style="margin-top: 0.25rem; color: #94a3b8; font-size: 0.9rem;">
                    All current readings match expected operational bounds. No unauthorized load detected.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Display recent 24-hour live chart for selected room
        df_room = load_all_readings(location_id=demo_room)
        recent_df = df_room.iloc[-48:].copy()

        fig_live = px.line(
            recent_df,
            x="timestamp",
            y="consumption_kwh",
            title=f"Telemetry Stream (Last 48 Hours) — {loc_map[demo_room]}",
            labels={"consumption_kwh": "Draw (kWh)", "timestamp": "Timestamp"}
        )
        fig_live.update_traces(line_color="#38bdf8", line_width=2.5)

        # Mark injected or anomaly points
        anom_pts = recent_df[recent_df["is_anomaly_flag"] == 1]
        if not anom_pts.empty:
            fig_live.add_trace(go.Scatter(
                x=anom_pts["timestamp"],
                y=anom_pts["consumption_kwh"],
                mode="markers",
                name="Flagged Anomaly / Test Event",
                marker=dict(color="#ef4444", size=10, symbol="x")
            ))

        fig_live.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0)",
            plot_bgcolor="rgba(30, 41, 59, 0.7)",
            height=340,
            margin=dict(l=20, r=20, t=35, b=20)
        )
        st.plotly_chart(fig_live, use_container_width=True)

    # Simulation Events Audit Log
    st.markdown("---")
    st.markdown("#### 📋 Simulation Event Audit Log")
    events = get_simulation_events(DEFAULT_DB_PATH, limit=10)
    if events:
        events_df = pd.DataFrame(events)[["timestamp", "location_name", "event_type", "consumption_kwh", "details"]]
        events_df.columns = ["Timestamp", "Location", "Event Type", "Reading (kWh)", "Audit Details"]
        st.dataframe(events_df, use_container_width=True)
    else:
        st.caption("No simulation events logged yet. Trigger a demo anomaly above to generate an audit trail.")


# ==============================================================================
# SECTION 3: ENERGY WASTE INTELLIGENCE ENGINE (FLAGSHIP FEATURE)
# ==============================================================================
elif nav_option == "🧠 Waste Intelligence Engine":
    st.subheader("🧠 Energy Waste Intelligence Engine")
    st.markdown("""
    **The flagship analytical core of WattWise AI.**  
    Combines machine learning anomaly scoring, physical room schedules, ambient weather, and room occupancy to distinguish **confirmed electricity waste** from **legitimate high operational load**.
    """)

    # Interactive Diagnostic Inspector
    st.markdown("#### 🔍 Real-Time Diagnostic Inspector")
    col_insp1, col_insp2, col_insp3, col_insp4 = st.columns(4)

    with col_insp1:
        insp_room = st.selectbox("Inspect Facility:", [l["id"] for l in locations], format_func=lambda x: loc_map[x])
    with col_insp2:
        insp_hour = st.slider("Time of Day (Hour):", 0, 23, 20, format="%d:00")
    with col_insp3:
        insp_occ = st.slider("Room Occupancy (People):", 0, 60, 0)
    with col_insp4:
        insp_kwh = st.slider("Measured Energy (kWh):", 0.1, 10.0, 4.8, step=0.1)

    # Perform evaluation
    loc_meta = loc_meta_map.get(insp_room, {})
    detector = EnergyAnomalyDetector(location_id=insp_room, contamination=anomaly_sensitivity)
    
    # Try loading trained detector
    detector.load()
    
    # Evaluate
    diag = detector.evaluate_single_reading(
        timestamp=f"2026-05-15 {insp_hour:02d}:00:00",
        consumption_kwh=insp_kwh,
        occupancy=insp_occ,
        temperature=26.0,
        tariff_per_kwh=active_tariff,
        operating_start=settings.get("operating_hours_start", 8),
        operating_end=settings.get("operating_hours_end", 18)
    )

    # Run through full Waste Intelligence Engine
    engine = EnergyWasteIntelligenceEngine(
        operating_start=settings.get("operating_hours_start", 8),
        operating_end=settings.get("operating_hours_end", 18),
        tariff_per_kwh=active_tariff
    )
    full_diag = engine.analyze_event(
        location_id=insp_room,
        location_name=loc_map[insp_room],
        timestamp=f"2026-05-15 {insp_hour:02d}:00:00",
        consumption_kwh=insp_kwh,
        expected_kwh=diag["expected_kwh"],
        occupancy=insp_occ,
        temperature=26.0,
        is_anomaly=diag["is_anomaly"],
        anomaly_score=diag["anomaly_score"],
        baseline_closed_kwh=loc_meta.get("baseline_kw_closed", 0.5),
        baseline_open_kwh=loc_meta.get("baseline_kw_open", 2.0)
    )

    # Diagnostic Output Display
    status_color = "#ef4444" if full_diag["is_waste"] else ("#38bdf8" if "OPERATIONAL" in full_diag["status"] else "#10b981")
    st.markdown(f"""
    <div style="background: #1e293b; border: 2px solid {status_color}; border-radius: 10px; padding: 1.5rem; margin-top: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h3 style="margin: 0; color: {status_color};">⚡ {full_diag['status']}</h3>
            <span style="background: rgba(255,255,255,0.1); padding: 0.3rem 0.8rem; border-radius: 6px; font-weight: 700; color: #f8fafc;">
                CONFIDENCE: {full_diag['confidence_pct']}%
            </span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem; padding: 1rem; background: #0f172a; border-radius: 8px;">
            <div>
                <span style="color: #94a3b8; font-size: 0.8rem;">MEASURED CONSUMPTION</span><br>
                <strong style="font-size: 1.4rem; color: #f1f5f9;">{full_diag['measured_kwh']:.2f} kWh</strong>
            </div>
            <div>
                <span style="color: #94a3b8; font-size: 0.8rem;">EXPECTED BASELINE</span><br>
                <strong style="font-size: 1.4rem; color: #38bdf8;">{full_diag['expected_kwh']:.2f} kWh</strong>
            </div>
            <div>
                <span style="color: #94a3b8; font-size: 0.8rem;">DEVIATION (DELTA)</span><br>
                <strong style="font-size: 1.4rem; color: {'#ef4444' if full_diag['diff_kwh'] > 0 else '#34d399'};">
                    {'+' if full_diag['diff_kwh'] > 0 else ''}{full_diag['diff_kwh']:.2f} kWh
                </strong>
            </div>
            <div>
                <span style="color: #94a3b8; font-size: 0.8rem;">FINANCIAL IMPACT</span><br>
                <strong style="font-size: 1.4rem; color: #fbbf24;">
                    +${full_diag['estimated_extra_cost']:.2f}
                </strong>
            </div>
        </div>
        <div style="margin-top: 1.25rem;">
            <strong style="color: #f8fafc; font-size: 1rem;">🔎 Diagnostic Insights & Plausible Causes:</strong>
            <ul style="margin-top: 0.5rem; color: #cbd5e1;">
                {''.join([f"<li>{p}</li>" for p in full_diag['possible_explanations']])}
            </ul>
        </div>
        <div style="margin-top: 0.75rem;">
            <strong style="color: #38bdf8; font-size: 1rem;">💡 Prescribed Action Plan:</strong>
            <ul style="margin-top: 0.5rem; color: #e2e8f0;">
                {''.join([f"<li>{a}</li>" for a in full_diag['recommended_actions']])}
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Waste Audit History on Dataset
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Comprehensive Historical Waste Audit")
    
    with st.spinner("Analyzing historical telemetry for energy waste patterns..."):
        df_audit = load_all_readings(location_id=insp_room)
        audit_res = engine.audit_dataset_for_waste(df_audit.iloc[-720:], locations_meta=loc_meta_map)

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.metric("Total Diagnosed Waste (Past 30 Days)", f"{audit_res['total_waste_kwh']:,.1f} kWh")
    with col_a2:
        st.metric("Financial Waste Impact", f"${audit_res['total_waste_cost']:,.2f}")
    with col_a3:
        st.metric("Flagged Waste Events", f"{audit_res['waste_events_count']} events")

    if audit_res["waste_events"]:
        audit_table = pd.DataFrame(audit_res["waste_events"])[
            ["timestamp", "location_name", "measured_kwh", "expected_kwh", "diff_kwh", "estimated_extra_cost", "status", "category"]
        ]
        audit_table.columns = ["Timestamp", "Location", "Measured (kWh)", "Expected (kWh)", "Excess (kWh)", "Cost Impact ($)", "Diagnostic Status", "Category"]
        st.dataframe(audit_table.tail(25), use_container_width=True)


# ==============================================================================
# SECTION: ENERGY WASTE DETECTIVE
# ==============================================================================
elif nav_option == "🕵️ Energy Waste Detective":
    st.subheader("🕵️ Energy Waste Detective — Evidence-Based Causal Diagnostics")
    st.markdown("""
    **Evidence-Based Waste Forensic Investigator:** Dissects unusual power surges by comparing observed consumption 
    against empirical baselines, formulating probabilistic root-cause hypotheses, calculating avoidable financial leakage, 
    and maintaining an audited investigation ledger.
    """)

    # Initialize detective instance with active user settings
    detective = WasteDetective(
        operating_start=int(settings.get("operating_hours_start", 8)),
        operating_end=int(settings.get("operating_hours_end", 18)),
        tariff_per_kwh=float(active_tariff)
    )

    # Auto-seed / sync historical anomalies into detective table if empty
    existing_findings = get_detective_findings(DEFAULT_DB_PATH, limit=10)
    if not existing_findings:
        with st.spinner("Analyzing historical telemetry to populate initial forensic findings..."):
            hist_readings = load_all_readings(location_id=None)
            if not hist_readings.empty:
                # Find top deviations and off-hours anomalies
                anom_subset = hist_readings[hist_readings["is_anomaly_flag"] == 1].tail(25)
                # Compute baselines
                baselines = hist_readings.groupby(["location_id", "hour"])["consumption_kwh"].median().to_dict()

                for _, a_row in anom_subset.iterrows():
                    l_id = a_row["location_id"]
                    l_name = loc_map.get(l_id, l_id)
                    hr = int(a_row["hour"])
                    base_kwh = baselines.get((l_id, hr), loc_meta_map.get(l_id, {}).get("baseline_kw_closed", 0.5))

                    finding = detective.generate_finding(
                        timestamp=str(a_row["timestamp"]),
                        location_id=l_id,
                        location_name=l_name,
                        observed_kwh=float(a_row["consumption_kwh"]),
                        baseline_kwh=float(base_kwh),
                        occupancy=int(a_row.get("occupancy", 0)),
                        temperature=float(a_row.get("temperature", 24.0)),
                        anomaly_score=float(a_row.get("anomaly_score", 0.75)),
                        is_synthetic_test=False
                    )
                    save_detective_finding(DEFAULT_DB_PATH, finding)

    # --------------------------------------------------------------------------
    # 1. Closed-Hours Demonstration Scenario Simulator
    # --------------------------------------------------------------------------
    with st.expander("🧪 **Reproducible Demo Scenario: Closed-Hours Surge Simulation**", expanded=False):
        st.markdown("""
        Test the real detection pipeline by creating a scenario where electricity consumption rises unexpectedly 
        during scheduled **closed hours** with zero occupants. All outputs below are **dynamically computed** by 
        the real detection pipeline (not hardcoded).
        """)

        sc_c1, sc_c2, sc_c3, sc_c4 = st.columns(4)
        with sc_c1:
            sc_room = st.selectbox("Scenario Facility:", [l["id"] for l in locations], format_func=lambda x: loc_map[x], key="sc_room")
        with sc_c2:
            sc_hour = st.slider("Off-Hours Time (Hour):", 19, 23, 22, format="%d:00", key="sc_hour")
        with sc_c3:
            sc_draw = st.slider("Observed Power Spike (kWh):", 2.0, 9.0, 5.4, step=0.1, key="sc_draw")
        with sc_c4:
            sc_occ = st.number_input("Occupants Present:", min_value=0, max_value=5, value=0, key="sc_occ", help="Simulating closed/vacant facility")

        if st.button("⚡ Run Scenario Through Real Detection Pipeline", type="primary", use_container_width=True):
            # Load ML detector for room if available
            detector = EnergyAnomalyDetector(location_id=sc_room, contamination=anomaly_sensitivity)
            detector.load()

            sc_time = datetime.now().strftime(f"%Y-%m-%d {sc_hour:02d}:30:00")
            base_kw = loc_meta_map.get(sc_room, {}).get("baseline_kw_closed", 0.8)

            scenario_payload = {
                "timestamp": sc_time,
                "location_id": sc_room,
                "location_name": loc_map[sc_room],
                "observed_kwh": sc_draw,
                "baseline_kwh": base_kw,
                "occupancy": sc_occ,
                "temperature": 26.5
            }

            # Run through actual detection and detective pipeline
            test_finding = detective.run_scenario_through_pipeline(
                scenario=scenario_payload,
                detector=detector,
                tariff_per_kwh=active_tariff
            )

            # Persist test finding into SQLite
            save_detective_finding(DEFAULT_DB_PATH, test_finding)

            # Also log to simulation events
            log_simulation_event(
                DEFAULT_DB_PATH,
                location_id=sc_room,
                event_type="DETECTIVE_SCENARIO_INJECTION",
                consumption_kwh=sc_draw,
                details=f"Closed-hours spike of {sc_draw:.2f} kWh tested through detective pipeline."
            )

            st.session_state.selected_finding_id = test_finding.get("timestamp")
            st.success("Scenario successfully routed through the real detection pipeline! See the explanation panel below.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 2. Forensic Findings Filter & Summary
    # --------------------------------------------------------------------------
    f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
    with f_col1:
        det_loc_filter = st.selectbox(
            "Filter Findings by Room:",
            ["all"] + [l["id"] for l in locations],
            format_func=lambda x: "All Facilities" if x == "all" else loc_map[x],
            key="det_loc"
        )
    with f_col2:
        det_status_filter = st.selectbox(
            "Filter by Investigation Status:",
            ["all", "PENDING", "INVESTIGATED", "CONFIRMED_WASTE", "FALSE_ALARM"],
            key="det_status"
        )
    with f_col3:
        # Fetch fresh findings from SQLite
        all_findings = get_detective_findings(
            DEFAULT_DB_PATH,
            location_id=det_loc_filter if det_loc_filter != "all" else None,
            status_filter=det_status_filter if det_status_filter != "all" else None,
            limit=100
        )
        csv_report = export_detective_findings_csv(all_findings)
        st.markdown("<div style='padding-top: 1.6rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Export Detective Findings (CSV)",
            data=csv_report,
            file_name="wattwise_detective_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Summary Metrics Row
    total_findings_cnt = len(all_findings)
    pending_cnt = sum(1 for f in all_findings if f["investigation_status"] == "PENDING")
    investigated_cnt = total_findings_cnt - pending_cnt
    total_leakage = sum(f["cost_impact"] for f in all_findings)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Forensic Findings", f"{total_findings_cnt} events")
    with kpi2:
        st.metric("Pending Investigation", f"{pending_cnt} unreviewed", delta=f"{pending_cnt} action required" if pending_cnt > 0 else "All reviewed", delta_color="inverse")
    with kpi3:
        st.metric("Investigated / Resolved", f"{investigated_cnt} closed")
    with kpi4:
        st.metric("Total Identified Leakage", f"${total_leakage:,.2f}", help="Sum of excess costs across all findings")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 3. Interactive Event Timeline
    # --------------------------------------------------------------------------
    st.markdown("#### ⏱️ Forensic Event Timeline")
    st.caption("Visual chronological timeline of suspicious consumption events. Dot color represents investigation status; size reflects financial impact.")

    if all_findings:
        df_timeline = pd.DataFrame(all_findings)
        df_timeline["timestamp_dt"] = pd.to_datetime(df_timeline["timestamp"])
        df_timeline["marker_size"] = np.clip(df_timeline["cost_impact"] * 18.0 + 8.0, 8.0, 32.0)

        color_map = {
            "PENDING": "#ef4444",           # Crimson red
            "INVESTIGATED": "#38bdf8",      # Sky blue
            "CONFIRMED_WASTE": "#f59e0b",   # Amber / Orange
            "FALSE_ALARM": "#94a3b8"        # Slate gray
        }

        fig_time = px.scatter(
            df_timeline,
            x="timestamp_dt",
            y="diff_kwh",
            color="investigation_status",
            size="marker_size",
            hover_data={
                "location_name": True,
                "observed_kwh": True,
                "baseline_kwh": True,
                "cost_impact": ":$.2f",
                "evidence_level": True,
                "investigation_status": True,
                "marker_size": False
            },
            color_discrete_map=color_map,
            labels={
                "timestamp_dt": "Event Timestamp",
                "diff_kwh": "Excess Draw Over Baseline (kWh)",
                "investigation_status": "Status"
            },
            title="Chronological Forensic Events & Excess Energy Deviation"
        )

        fig_time.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0)",
            plot_bgcolor="rgba(30, 41, 59, 0.7)",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No forensic findings meet the current filter criteria.")

    # --------------------------------------------------------------------------
    # 4. Detailed Forensic Explanation Panel & Investigation Action
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🔎 Finding Forensic Explanation & Auditor Action Panel")

    if all_findings:
        # Finding selector dropdown
        finding_options = [
            f"{f['id']} | {f['timestamp']} | {f['location_name']} | +{f['diff_kwh']:.2f} kWh (${f['cost_impact']:.2f}) | [{f['investigation_status']}]"
            for f in all_findings
        ]
        selected_idx = st.selectbox(
            "Select Finding to Inspect:",
            range(len(finding_options)),
            format_func=lambda i: finding_options[i]
        )
        chosen = all_findings[selected_idx]

        # Check if synthetic test
        test_badge = (
            "<span style='background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid #c084fc; "
            "padding: 0.2rem 0.6rem; border-radius: 9999px; font-weight: 700; font-size: 0.75rem;'>"
            "🧪 SYNTHETIC TEST EVENT</span>"
            if chosen.get("is_synthetic_test") == 1 else
            "<span style='background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #34d399; "
            "padding: 0.2rem 0.6rem; border-radius: 9999px; font-weight: 700; font-size: 0.75rem;'>"
            "📊 SYNTHETIC BASELINE TELEMETRY</span>"
        )

        status_class = (
            "status-badge-danger" if chosen["investigation_status"] == "PENDING" else
            ("status-badge-warn" if chosen["investigation_status"] == "CONFIRMED_WASTE" else "status-badge-ok")
        )

        # Main explanation card
        st.markdown(f"""
        <div style="background: #1e293b; border-radius: 12px; border: 1px solid #334155; padding: 1.5rem; margin-top: 0.75rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 0.75rem;">
                <div>
                    <h3 style="margin: 0; color: #f8fafc;">⚡ Forensic Finding #{chosen['id']} — {chosen['location_name']}</h3>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem;">Timestamp: {chosen['timestamp']} | Recorded Hour: {chosen['hour']:02d}:00</div>
                </div>
                <div>
                    {test_badge}
                    <span class="{status_class}" style="margin-left: 0.5rem;">STATUS: {chosen['investigation_status']}</span>
                </div>
            </div>

            <!-- Quantitative Telemetry Cards -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1.25rem; background: #0f172a; padding: 1.25rem; border-radius: 8px;">
                <div>
                    <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Observed Consumption</span><br>
                    <strong style="font-size: 1.5rem; color: #f1f5f9;">{chosen['observed_kwh']:.2f} kWh</strong>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Baseline Expectation</span><br>
                    <strong style="font-size: 1.5rem; color: #38bdf8;">{chosen['baseline_kwh']:.2f} kWh</strong>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Calculated Delta</span><br>
                    <strong style="font-size: 1.5rem; color: #ef4444;">+{chosen['diff_kwh']:.2f} kWh</strong>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Avoidable Cost Leakage</span><br>
                    <strong style="font-size: 1.5rem; color: #fbbf24;">${chosen['cost_impact']:.2f}</strong>
                    <div style="font-size: 0.75rem; color: #64748b;">@ ${chosen['tariff_rate']:.2f}/kWh</div>
                </div>
            </div>

            <!-- Why Flagged & Evidence -->
            <div style="margin-top: 1.25rem; padding: 1rem; background: rgba(30, 41, 59, 0.5); border-left: 4px solid #38bdf8; border-radius: 4px;">
                <strong style="color: #38bdf8; font-size: 0.95rem;">🚩 Why This Observation Was Flagged:</strong>
                <p style="margin: 0.4rem 0 0 0; color: #e2e8f0; font-size: 0.92rem;">{chosen['flag_reason']}</p>
                <div style="margin-top: 0.5rem; color: #94a3b8; font-size: 0.82rem;">
                    <strong>Evidence Confidence Indicator:</strong> <span style="color: #34d399; font-weight: 700;">{chosen['evidence_level']} ({int(chosen['evidence_score'] * 100)}%)</span>
                </div>
            </div>

            <!-- Plausible Causes -->
            <div style="margin-top: 1.25rem;">
                <strong style="color: #f8fafc; font-size: 0.95rem;">🔬 Plausible Contributing Factors (Probabilistic Hypotheses):</strong>
                <ul style="margin: 0.5rem 0 0.5rem 1.25rem; color: #cbd5e1; font-size: 0.9rem;">
                    {''.join([f"<li style='margin-bottom: 0.35rem;'>{cause}</li>" for cause in chosen['plausible_causes_list']])}
                </ul>
                <div style="font-size: 0.75rem; color: #64748b; font-style: italic;">
                    *Note: Plausible causes represent hypothesis formulations based on room category, schedule, and load profile, without claiming absolute physical certainty.
                </div>
            </div>

            <!-- Recommended Practical Action -->
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px;">
                <strong style="color: #34d399; font-size: 0.95rem;">💡 Recommended Practical Action:</strong>
                <div style="margin-top: 0.35rem; color: #f1f5f9; font-size: 0.92rem;">{chosen['recommended_action']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # 5. Interactive Investigation Workflow Form (Saves to SQLite)
        # ----------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📝 Auditor Investigation Workbench")

        with st.form(key=f"inv_form_{chosen['id']}"):
            f_col_stat, f_col_note = st.columns([1, 2])
            with f_col_stat:
                new_status = st.selectbox(
                    "Update Investigation Status:",
                    ["PENDING", "INVESTIGATED", "CONFIRMED_WASTE", "FALSE_ALARM"],
                    index=["PENDING", "INVESTIGATED", "CONFIRMED_WASTE", "FALSE_ALARM"].index(chosen["investigation_status"])
                )
            with f_col_note:
                current_notes = chosen.get("investigator_notes", "")
                new_notes = st.text_area(
                    "Auditor Field Notes & Resolution Summary:",
                    value=current_notes,
                    placeholder="Enter on-site verification findings (e.g., 'Security confirmed 25 PCs active without users; powered off at 22:45.').",
                    height=100
                )

            submit_save = st.form_submit_button("💾 Save Investigation Status to SQLite", type="primary", use_container_width=True)

            if submit_save:
                updated = update_investigation_status(
                    DEFAULT_DB_PATH,
                    finding_id=chosen["id"],
                    new_status=new_status,
                    investigator_notes=new_notes
                )
                if updated:
                    st.toast(f"✅ Finding #{chosen['id']} updated to {new_status}!", icon="💾")
                    st.success(f"Investigation record #{chosen['id']} successfully persisted to SQLite.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to update record in database.")


# ==============================================================================
# SECTION 4: AI FORECASTING
# ==============================================================================
elif nav_option == "📈 AI Forecasting":
    st.subheader("📈 AI Electricity Demand Forecasting")
    st.markdown("""
    Predictive load forecasting using Scikit-Learn **RandomForestRegressor**.  
    Trained strictly with a chronological train/test split to avoid lookahead bias and evaluated against an empirical **Hourly Baseline Model**.
    """)

    fc_room = st.selectbox("Select Target Facility for Forecasting:", [l["id"] for l in locations], format_func=lambda x: loc_map[x])
    df_loc = load_all_readings(location_id=fc_room)

    forecaster = ElectricityForecaster(location_id=fc_room)
    has_model = forecaster.load()

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        if st.button("🔄 Retrain Forecasting Model", use_container_width=True):
            with st.spinner(f"Training RandomForest model on {len(df_loc):,} records..."):
                metrics = forecaster.train_and_evaluate(df_loc, test_fraction=0.2)
                st.success("Model trained and serialized successfully!")
                has_model = True

    if not has_model:
        st.info("No serialized model found for this location. Click 'Retrain Forecasting Model' to train.")
    else:
        metrics = forecaster.metrics
        st.markdown("#### 🎯 Evaluation Metrics on Held-Out Test Set")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        rf_mae = metrics.get("random_forest", {}).get("mae", 0.0)
        rf_rmse = metrics.get("random_forest", {}).get("rmse", 0.0)
        base_mae = metrics.get("baseline_model", {}).get("mae", 0.0)
        base_rmse = metrics.get("baseline_model", {}).get("rmse", 0.0)
        impr = metrics.get("improvement_pct", {}).get("mae", 0.0)

        with m_col1:
            st.metric("Random Forest MAE", f"{rf_mae:.4f} kWh", help="Mean Absolute Error on test split")
        with m_col2:
            st.metric("Baseline Model MAE", f"{base_mae:.4f} kWh", help="Empirical hourly mean baseline MAE")
        with m_col3:
            st.metric("Random Forest RMSE", f"{rf_rmse:.4f} kWh")
        with m_col4:
            st.metric("Accuracy Improvement", f"+{impr:.1f}%", delta=f"{impr:.1f}% vs Baseline")

        # Test Set: Actual vs Predicted Plot
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔬 Test Set Horizon: Actual vs Predicted Load")
        
        sample_ts = metrics.get("test_timestamps_sample", [])
        sample_actual = metrics.get("test_actual_sample", [])
        sample_pred = metrics.get("test_predicted_sample", [])

        if sample_ts and sample_actual and sample_pred:
            fig_eval = go.Figure()
            fig_eval.add_trace(go.Scatter(
                x=sample_ts, y=sample_actual,
                name="Actual Consumption (kWh)",
                line=dict(color="#38bdf8", width=2)
            ))
            fig_eval.add_trace(go.Scatter(
                x=sample_ts, y=sample_pred,
                name="Random Forest Prediction (kWh)",
                line=dict(color="#10b981", width=2.5, dash="dash")
            ))
            fig_eval.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0)",
                plot_bgcolor="rgba(30, 41, 59, 0.7)",
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", y=1.05)
            )
            st.plotly_chart(fig_eval, use_container_width=True)

        # Forward Next-24-Hour Forecast
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔮 Next-24-Hour Forward Forecast")
        
        future_df = forecaster.predict_next_24h(
            expected_operating_start=loc_meta_map.get(fc_room, {}).get("operating_start", 8),
            expected_operating_end=loc_meta_map.get(fc_room, {}).get("operating_end", 18)
        )

        fig_fut = go.Figure()
        # Upper/lower bounds
        fig_fut.add_trace(go.Scatter(
            x=future_df["timestamp"], y=future_df["upper_bound_kwh"],
            mode="lines", line=dict(width=0), showlegend=False
        ))
        fig_fut.add_trace(go.Scatter(
            x=future_df["timestamp"], y=future_df["lower_bound_kwh"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(16, 185, 129, 0.15)",
            name="95% Confidence Interval"
        ))
        # Forecast line
        fig_fut.add_trace(go.Scatter(
            x=future_df["timestamp"], y=future_df["predicted_kwh"],
            mode="lines+markers", line=dict(color="#10b981", width=3),
            name="Projected Load (kWh)"
        ))
        fig_fut.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0)",
            plot_bgcolor="rgba(30, 41, 59, 0.7)",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(title="Projected kWh"),
            xaxis=dict(title="Future 24-Hour Horizon")
        )
        st.plotly_chart(fig_fut, use_container_width=True)

        st.caption("⚠️ **Forecast Transparency Note:** Projections are generated based on historical cyclic patterns, scheduled room hours, and expected weather. Forecasts do not account for unannounced events or power outages.")


# ==============================================================================
# SECTION 5: ENERGY SAVINGS LAB (WHAT-IF SIMULATOR)
# ==============================================================================
elif nav_option == "🔬 Energy Savings Lab":
    st.subheader("🔬 Energy Savings Lab (What-If Simulator)")
    st.markdown("""
    **Interactive Conservation Modeler:** Simulate operational schedule reductions, hardware efficiency retrofits (e.g. LED tubes, Mini-PCs, BLDC fans), and calculate verified financial ROI and carbon offsets.
    """)

    st.markdown("#### ⚡ Select Benchmark Preset or Customize Parameters")
    preset_col, _ = st.columns([2, 1])
    with preset_col:
        preset_choice = st.selectbox(
            "Load Scenario Preset:",
            [
                ("computer_lab_baseline", "Computer Lab: Standard Desktops vs Eco Mini-PCs"),
                ("classroom_baseline", "Classroom: Conventional Fluorescent vs LED & BLDC")
            ],
            format_func=lambda x: x[1]
        )[0]

    base_preset = get_preset_scenario(preset_choice)
    prop_preset = get_preset_scenario(preset_choice.replace("baseline", "retrofitted"))

    col_base, col_prop = st.columns(2)

    with col_base:
        st.markdown("### 🏢 Baseline Scenario (Current)")
        b_comp_qty = st.number_input("Number of Computers:", 0, 100, base_preset["num_computers"], key="b_comp_q")
        b_comp_w = st.number_input("Computer Power (Watts):", 10.0, 500.0, base_preset["computer_watts"], key="b_comp_w")
        b_comp_h = st.slider("Computer Daily Hours:", 0.0, 24.0, base_preset["computer_hours"], step=0.5, key="b_comp_h")

        b_light_qty = st.number_input("Number of Light Fixtures:", 0, 100, base_preset["num_lights"], key="b_light_q")
        b_light_w = st.number_input("Light Fixture Power (Watts):", 5.0, 200.0, base_preset["light_watts"], key="b_light_w")
        b_light_h = st.slider("Lighting Daily Hours:", 0.0, 24.0, base_preset["light_hours"], step=0.5, key="b_light_h")

        b_fan_qty = st.number_input("Number of Fans / AC Units:", 0, 50, base_preset["num_fans"], key="b_fan_q")
        b_fan_w = st.number_input("Fan / AC Power (Watts):", 10.0, 2000.0, base_preset["fan_watts"], key="b_fan_w")
        b_fan_h = st.slider("Cooling Daily Hours:", 0.0, 24.0, base_preset["fan_hours"], step=0.5, key="b_fan_h")

    with col_prop:
        st.markdown("### 🌱 Proposed Scenario (Optimized)")
        p_comp_qty = st.number_input("Number of Computers:", 0, 100, prop_preset["num_computers"], key="p_comp_q")
        p_comp_w = st.number_input("Computer Power (Watts):", 10.0, 500.0, prop_preset["computer_watts"], key="p_comp_w")
        p_comp_h = st.slider("Computer Daily Hours:", 0.0, 24.0, prop_preset["computer_hours"], step=0.5, key="p_comp_h")

        p_light_qty = st.number_input("Number of Light Fixtures:", 0, 100, prop_preset["num_lights"], key="p_light_q")
        p_light_w = st.number_input("Light Fixture Power (Watts):", 5.0, 200.0, prop_preset["light_watts"], key="p_light_w")
        p_light_h = st.slider("Lighting Daily Hours:", 0.0, 24.0, prop_preset["light_hours"], step=0.5, key="p_light_h")

        p_fan_qty = st.number_input("Number of Fans / AC Units:", 0, 50, prop_preset["num_fans"], key="p_fan_q")
        p_fan_w = st.number_input("Fan / AC Power (Watts):", 10.0, 2000.0, prop_preset["fan_watts"], key="p_fan_w")
        p_fan_h = st.slider("Cooling Daily Hours:", 0.0, 24.0, prop_preset["fan_hours"], step=0.5, key="p_fan_h")

    retrofit_capex = st.number_input("Optional Retrofit Investment Cost ($):", min_value=0.0, value=850.0, step=50.0)

    # Perform Simulation
    base_dict = {
        "num_computers": b_comp_qty, "computer_watts": b_comp_w, "computer_hours": b_comp_h,
        "num_lights": b_light_qty, "light_watts": b_light_w, "light_hours": b_light_h,
        "num_fans": b_fan_qty, "fan_watts": b_fan_w, "fan_hours": b_fan_h
    }
    prop_dict = {
        "num_computers": p_comp_qty, "computer_watts": p_comp_w, "computer_hours": p_comp_h,
        "num_lights": p_light_qty, "light_watts": p_light_w, "light_hours": p_light_h,
        "num_fans": p_fan_qty, "fan_watts": p_fan_w, "fan_hours": p_fan_h
    }

    sim_res = simulate_room_savings(
        baseline_params=base_dict,
        proposed_params=prop_dict,
        tariff_per_kwh=active_tariff,
        retrofit_cost=retrofit_capex
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💰 Projected Energy & Cost Reductions")

    sav = sim_res["savings"]
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1:
        st.metric("Monthly Savings", f"{sav['monthly_kwh']:,.1f} kWh", delta=f"-{sav['percent_reduction']}%")
    with sc2:
        st.metric("Monthly Bill Reduction", f"${sav['monthly_cost']:,.2f}")
    with sc3:
        st.metric("Annual Savings", f"${sav['annual_cost']:,.2f}")
    with sc4:
        st.metric("CO2 Avoided / Year", f"{sav['annual_co2_kg']:,.0f} kg", help="Based on 0.82 kg CO2 / kWh")
    with sc5:
        payback_str = f"{sav['payback_months']} Months" if sav['payback_months'] < 100 else "N/A"
        st.metric("Simple Payback", payback_str)

    # Comparison Bar Chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Comparative Breakdown by Appliance Group")

    app_df = pd.DataFrame({
        "Category": ["Computers", "Lighting", "Cooling/Fans"],
        "Baseline (kWh/day)": [
            sim_res["baseline"]["appliance_breakdown_daily_kwh"]["computers"],
            sim_res["baseline"]["appliance_breakdown_daily_kwh"]["lights"],
            sim_res["baseline"]["appliance_breakdown_daily_kwh"]["fans"]
        ],
        "Proposed (kWh/day)": [
            sim_res["proposed"]["appliance_breakdown_daily_kwh"]["computers"],
            sim_res["proposed"]["appliance_breakdown_daily_kwh"]["lights"],
            sim_res["proposed"]["appliance_breakdown_daily_kwh"]["fans"]
        ]
    })

    fig_comp = go.Figure(data=[
        go.Bar(name="Baseline", x=app_df["Category"], y=app_df["Baseline (kWh/day)"], marker_color="#f87171"),
        go.Bar(name="Proposed (Eco-Optimized)", x=app_df["Category"], y=app_df["Proposed (kWh/day)"], marker_color="#34d399")
    ])
    fig_comp.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0)",
        plot_bgcolor="rgba(30, 41, 59, 0.7)",
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(title="Daily Consumption (kWh)")
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.caption("ℹ️ **Simulation Notice:** All metrics reflect theoretical physics-based models using: `Energy (kWh) = (Watts × Quantity × Hours) / 1000`. Actual savings may vary depending on power factor, ambient thermal fluctuations, and occupant usage behavior.")


# ==============================================================================
# SECTION 6: ROOM COMPARISON
# ==============================================================================
elif nav_option == "🏢 Room Comparison":
    st.subheader("🏢 Multi-Facility Energy Comparison & Heatmaps")
    st.markdown("Compare energy intensity, peak demand hours, and efficiency scores across all monitored rooms.")

    df_all = load_all_readings(location_id=None)

    if not df_all.empty:
        # Aggregation by location
        summary_df = df_all.groupby(["location_id", "location_name"]).agg(
            total_kwh=("consumption_kwh", "sum"),
            avg_hourly_kwh=("consumption_kwh", "mean"),
            peak_kwh=("consumption_kwh", "max"),
            anomalies=("is_anomaly_flag", "sum")
        ).reset_index()

        summary_df["estimated_cost"] = (summary_df["total_kwh"] * active_tariff).round(2)
        summary_df["total_kwh"] = summary_df["total_kwh"].round(1)
        summary_df["avg_hourly_kwh"] = summary_df["avg_hourly_kwh"].round(2)

        st.dataframe(
            summary_df[["location_name", "total_kwh", "estimated_cost", "avg_hourly_kwh", "peak_kwh", "anomalies"]].rename(columns={
                "location_name": "Facility",
                "total_kwh": "Total Energy (kWh)",
                "estimated_cost": "Total Cost ($)",
                "avg_hourly_kwh": "Avg Hourly (kWh)",
                "peak_kwh": "Peak Demand (kW)",
                "anomalies": "Flagged Anomalies"
            }),
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            fig_pie = px.pie(
                summary_df,
                values="total_kwh",
                names="location_name",
                title="Electricity Consumption Distribution by Facility",
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0)",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_c2:
            # Diurnal heatmap: Hour of Day vs Location
            df_heat = df_all.copy()
            heat_pivot = df_heat.pivot_table(
                index="location_name",
                columns="hour",
                values="consumption_kwh",
                aggfunc="mean"
            ).round(2)

            fig_heat = px.imshow(
                heat_pivot,
                labels=dict(x="Hour of Day (00:00 - 23:00)", y="Location", color="Mean kWh"),
                title="Hourly Consumption Heatmap (Average Demand Profile)",
                color_continuous_scale="Viridis"
            )
            fig_heat.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0)",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_heat, use_container_width=True)


# ==============================================================================
# SECTION 7: ELECTRICITY BILL ESTIMATOR
# ==============================================================================
elif nav_option == "💵 Electricity Bill Estimator":
    st.subheader("💵 Transparent Electricity Bill Estimator")
    st.markdown("""
    Calculate itemized utility charges based on active energy consumption, fixed connection fees, municipal taxes, and peak demand surcharges.
    """)

    b_col1, b_col2 = st.columns([1, 1])

    with b_col1:
        st.markdown("#### ⚙️ Utility Tariff Settings")
        bill_kwh = st.number_input("Monthly Consumption (kWh):", min_value=0.0, value=2500.0, step=100.0)
        bill_rate = st.number_input("Base Energy Tariff ($/kWh):", min_value=0.01, value=float(active_tariff), step=0.01)
        bill_fixed = st.number_input("Fixed Monthly Service Charge ($):", min_value=0.0, value=float(settings.get("fixed_monthly_charge", 25.0)), step=5.0)
        bill_tax = st.slider("Tax / Surcharge Rate (%):", 0.0, 20.0, float(settings.get("tax_surcharge_rate", 0.05)) * 100.0, step=0.5) / 100.0
        bill_peak = st.number_input("Peak Demand (kW):", min_value=0.0, value=12.5, step=0.5)
        bill_demand_rate = st.number_input("Demand Charge ($/kW):", min_value=0.0, value=3.5, step=0.5)
        prev_kwh = st.number_input("Previous Month Consumption (kWh):", min_value=0.0, value=2800.0, step=100.0)

    bill_calc = calculate_electricity_bill(
        total_kwh=bill_kwh,
        tariff_per_kwh=bill_rate,
        fixed_monthly_charge=bill_fixed,
        tax_surcharge_rate=bill_tax,
        peak_kw=bill_peak,
        demand_charge_per_kw=bill_demand_rate,
        previous_period_kwh=prev_kwh
    )

    with b_col2:
        st.markdown("#### 🧾 Itemized Bill Breakdown")
        charges = bill_calc["itemized_charges"]
        
        st.markdown(f"""
        <div style="background: #1e293b; border-radius: 10px; padding: 1.5rem; border: 1px solid #334155;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 0.5rem;">
                <span style="color: #94a3b8;">Volumetric Energy Charge ({bill_calc['total_kwh']} kWh @ ${bill_rate:.2f})</span>
                <strong style="color: #f8fafc;">${charges['energy_charge']:.2f}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding: 0.5rem 0;">
                <span style="color: #94a3b8;">Fixed Monthly Connection Charge</span>
                <strong style="color: #f8fafc;">${charges['fixed_charge']:.2f}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding: 0.5rem 0;">
                <span style="color: #94a3b8;">Peak Demand Charge ({bill_peak} kW @ ${bill_demand_rate:.2f})</span>
                <strong style="color: #f8fafc;">${charges['demand_charge']:.2f}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding: 0.5rem 0;">
                <span style="color: #94a3b8;">Subtotal</span>
                <strong style="color: #38bdf8;">${charges['subtotal']:.2f}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #38bdf8; padding: 0.5rem 0;">
                <span style="color: #94a3b8;">Taxes & Surcharges ({charges['tax_rate_percent']}%)</span>
                <strong style="color: #f8fafc;">${charges['tax_and_surcharge']:.2f}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; padding-top: 1rem;">
                <span style="font-size: 1.2rem; font-weight: 800; color: #f8fafc;">TOTAL ESTIMATED BILL</span>
                <span style="font-size: 1.5rem; font-weight: 800; color: #34d399;">${bill_calc['total_bill']:.2f}</span>
            </div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">
                Effective All-In Rate: ${bill_calc['effective_rate_per_kwh']:.4f} per kWh
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Variance with previous period
        cmp = bill_calc["comparison"]
        if cmp:
            trend_color = "#34d399" if cmp["cost_difference"] < 0 else "#f87171"
            st.markdown(f"""
            <div style="margin-top: 1rem; background: #0f172a; padding: 1rem; border-radius: 8px; border: 1px solid #334155;">
                <span style="color: #94a3b8; font-size: 0.85rem;">MONTH-OVER-MONTH VARIANCE</span><br>
                <strong style="color: {trend_color}; font-size: 1.1rem;">
                    {cmp['trend']}: {'+' if cmp['cost_difference'] > 0 else ''}${cmp['cost_difference']:.2f} ({'+' if cmp['percent_change'] > 0 else ''}{cmp['percent_change']}%)
                </strong>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.25rem;">
                    Prior Month: {cmp['previous_kwh']} kWh (${cmp['previous_total_bill']:.2f})
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 8: REPORTS, EXPORT & SETTINGS
# ==============================================================================
elif nav_option == "📑 Reports & Settings":
    st.subheader("📑 Telemetry Export, Audit Reporting & System Settings")

    tab_reports, tab_recs, tab_settings = st.tabs(["📥 Data Export & Executive Report", "💡 Intelligent Recommendations", "⚙️ Database & App Settings"])

    df_export = load_all_readings(location_id=current_loc_filter)

    with tab_reports:
        st.markdown("#### 📥 CSV Telemetry Exports")
        
        c_exp1, c_exp2, c_exp3 = st.columns(3)
        with c_exp1:
            csv_data = export_consumption_csv(df_export)
            st.download_button(
                "⬇️ Download Consumption Data (CSV)",
                data=csv_data,
                file_name=f"wattwise_consumption_{current_loc_filter}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c_exp2:
            csv_anoms = export_anomalies_csv(df_export)
            st.download_button(
                "⬇️ Download Anomaly Audit (CSV)",
                data=csv_anoms,
                file_name=f"wattwise_anomalies_{current_loc_filter}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c_exp3:
            rec_engine = RecommendationEngine(tariff_per_kwh=active_tariff)
            recs_list = rec_engine.generate_recommendations(df_export, locations_meta=loc_meta_map)
            csv_recs = export_recommendations_csv(recs_list)
            st.download_button(
                "⬇️ Download Recommendations (CSV)",
                data=csv_recs,
                file_name="wattwise_recommendations.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📄 Executive Energy Audit Briefing")
        
        total_kwh = float(df_export["consumption_kwh"].sum()) if not df_export.empty else 0.0
        total_cost = total_kwh * active_tariff
        anom_cnt = int(df_export["is_anomaly_flag"].sum()) if not df_export.empty else 0
        peak_draw = float(df_export["consumption_kwh"].max()) if not df_export.empty else 0.0

        exec_report = generate_executive_summary_markdown(
            location_name=loc_map.get(current_loc_filter, "All Monitored Facilities"),
            date_range_str="180-Day Continuous Historical Telemetry",
            total_kwh=total_kwh,
            total_cost=total_cost,
            peak_demand_kw=peak_draw,
            anomalies_count=anom_cnt,
            waste_kwh=anom_cnt * 2.8,
            waste_cost=(anom_cnt * 2.8) * active_tariff,
            forecast_next_24h_kwh=65.4,
            actionable_recs_count=len(recs_list),
            currency_symbol="$"
        )

        st.markdown(exec_report)
        st.download_button(
            "⬇️ Download Executive Audit Report (.MD)",
            data=exec_report,
            file_name="wattwise_executive_report.md",
            mime="text/markdown"
        )

    with tab_recs:
        st.markdown("#### 💡 Intelligent Energy Optimization Recommendations")
        rec_engine = RecommendationEngine(tariff_per_kwh=active_tariff)
        live_recs = rec_engine.generate_recommendations(df_export, locations_meta=loc_meta_map)

        if not live_recs:
            st.info("No active high-priority recommendations detected for current filter.")
        else:
            for r in live_recs:
                conf_badge = "status-badge-ok" if r["confidence_level"] == "High" else "status-badge-warn"
                st.markdown(f"""
                <div style="background: #1e293b; border-radius: 8px; border: 1px solid #334155; padding: 1.25rem; margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #f8fafc;">💡 {r['title']}</h4>
                        <span class="{conf_badge}">Confidence: {r['confidence_level']}</span>
                    </div>
                    <div style="margin-top: 0.5rem; color: #94a3b8; font-size: 0.9rem;">{r['reason']}</div>
                    <div style="margin-top: 0.5rem; background: #0f172a; padding: 0.75rem; border-radius: 6px; color: #cbd5e1; font-size: 0.85rem;">
                        <strong>Telemetry Evidence:</strong> {r['supporting_data']}
                    </div>
                    <div style="margin-top: 0.5rem; color: #38bdf8; font-size: 0.9rem;">
                        <strong>Prescribed Action:</strong> {r['suggested_action']}
                    </div>
                    <div style="margin-top: 0.75rem; font-weight: 700; color: #34d399; font-size: 0.95rem;">
                        Potential Recovery: ~{r['estimated_savings_kwh']} kWh/month (~${r['estimated_savings_cost']:.2f}/month)
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_settings:
        st.markdown("#### ⚙️ Global Facility Settings & Persistence")
        
        c_set1, c_set2 = st.columns(2)
        with c_set1:
            new_op_start = st.number_input("Scheduled Opening Hour (24h):", 0, 23, int(settings.get("operating_hours_start", 8)))
            new_op_end = st.number_input("Scheduled Closing Hour (24h):", 0, 23, int(settings.get("operating_hours_end", 18)))
            new_co2 = st.number_input("Carbon Factor (kg CO2 / kWh):", 0.1, 2.0, float(settings.get("co2_factor_kg_per_kwh", 0.82)), step=0.01)

        with c_set2:
            new_curr = st.text_input("Currency Symbol:", str(settings.get("currency_symbol", "$")))
            new_fixed = st.number_input("Default Fixed Monthly Fee ($):", 0.0, 500.0, float(settings.get("fixed_monthly_charge", 25.0)), step=5.0)

        if st.button("💾 Save Settings to Database", type="primary"):
            set_setting(DEFAULT_DB_PATH, "operating_hours_start", new_op_start)
            set_setting(DEFAULT_DB_PATH, "operating_hours_end", new_op_end)
            set_setting(DEFAULT_DB_PATH, "co2_factor_kg_per_kwh", new_co2)
            set_setting(DEFAULT_DB_PATH, "currency_symbol", new_curr)
            set_setting(DEFAULT_DB_PATH, "fixed_monthly_charge", new_fixed)
            st.success("Settings updated successfully!")
            time.sleep(1)
            st.rerun()

        st.markdown("---")
        st.markdown("#### ⚠️ Danger Zone")
        if st.button("🧹 Reset Demo Injected Events & Audit Logs"):
            reset_demo_data(DEFAULT_DB_PATH)
            st.success("Demo anomalies and simulation logs cleared. Pure baseline restored.")
            time.sleep(1)
            st.rerun()

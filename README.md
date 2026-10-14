# ⚡ WattWise AI
> **"Detect Waste. Predict Demand. Save Energy."**

An intelligent, AI-powered electricity management assistant designed specifically for **college computer laboratories, classrooms, libraries, faculty offices, and smart facilities**. 

WattWise AI empowers administrators and students to detect after-hours energy waste, forecast future load demands using machine learning, simulate "what-if" efficiency retrofits, and calculate transparent electricity bills—all without requiring expensive physical IoT sensors or paid proprietary cloud APIs.

---

## 🌟 Unique Core Features

### 1. 🧠 Energy Waste Intelligence Engine (Flagship Feature)
Unlike naive anomaly detectors that trigger false alarms during legitimate high-occupancy events, the **Energy Waste Intelligence Engine** combines:
- Machine learning anomaly scores from **IsolationForest**
- Scheduled facility operating hours (e.g., 08:00 – 18:00)
- Real-time occupant headcounts
- Diurnal ambient weather fluctuations
- Historical hourly median baselines

It classifies events into actionable categories:
- **Confirmed Electricity Waste:** High power draw during closed hours with 0 occupancy (e.g. computer lab workstations or AC units left running overnight).
- **Unoccupied Daytime Waste:** Empty lecture rooms with lights and projectors running.
- **High Operational Demand (Not Waste):** High daytime draw that legitimately correlates with a crowded lab or workshop.
- **Abnormal Hardware Spikes:** Equipment electrical faults or unauthorized appliances.

Every flagged waste event provides:
- Measured vs. Expected consumption (kWh)
- Avoidable financial leakage ($)
- Multi-factor root cause diagnosis
- Recommended next operational action

### 2. 🕵️ Energy Waste Detective (Evidence-Based Causal Forensics)
A specialized forensic workbench for energy waste investigations:
- **Evidence-Based Explanations:** Compares observed consumption with an empirical baseline, calculates $\Delta\text{kWh}$ and avoidable dollar cost at the configured tariff.
- **Probabilistic Hypotheses:** Formulates plausible contributing causes (e.g. setback overrides, unattended workstation clusters, equipment faults) without claiming false certainty.
- **Audited Investigation Workflow:** Allows facility managers to update finding status (`PENDING`, `INVESTIGATED`, `CONFIRMED_WASTE`, `FALSE_ALARM`) and attach field notes stored persistently in SQLite.
- **Interactive Event Timeline:** Color-coded chronological timeline mapping excess power draw against time.
- **Explanation Panel:** Detailed breakdown of baseline vs. observed draw, evidence indicators, and recommended actions.
- **Closed-Hours Test Scenario:** Reproducible simulation of off-hours power spikes routed dynamically through the detection pipeline (clearly distinguished as synthetic test events).
- **Forensic CSV Export:** 1-click download of all investigation findings and auditor notes.

### 3. 🔬 Energy Savings Lab (What-If Simulator)
An interactive physics-based conservation modeling lab:
- Adjust operating hours, computer counts, computer wattages (e.g., 150W desktop vs. 45W mini-PC), lighting fixtures (40W fluorescent vs. 14W LED), and cooling fans (75W standard vs. 35W BLDC).
- Deterministic calculation using:
  $$\text{Energy (kWh)} = \frac{\sum (\text{Quantity} \times \text{Power in Watts} \times \text{Operating Hours})}{1000}$$
- Instant outputs: Daily & Monthly kWh savings, percentage reduction, financial savings, carbon emissions avoided ($\text{kg CO}_2$), and simple retrofit payback period (months).

### 3. 📈 AI Electricity Demand Forecasting
- Powered by Scikit-Learn **RandomForestRegressor**.
- Trained using a **strictly chronological train/test split** (earlier 80% train, held-out 20% test) to prevent future data leakage.
- Fairly benchmarked against an empirical **Hourly Baseline Model**.
- Displays test set Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and percentage accuracy improvement.
- Generates a forward **Next-24-Hour Horizon Forecast** with 95% confidence intervals based on upcoming calendar and diurnal assumptions.

### 4. 🚨 Live Demo Mode & Anomaly Injection
Built for live college presentations and judge evaluations:
- Continuous simulated live telemetry stream.
- **"Trigger Demo Anomaly" Button:** Injects a reproducible real-time off-hours power spike directly into the database and routes it through the live detection and waste intelligence pipeline.
- Marked clearly as a test event with full audit logging.
- **"Reset Demo State" Button:** Restores pure synthetic baseline data with one click.

### 5. 🏢 Multi-Room Facility Monitoring
Pre-configured with distinct synthetic profiles for 5 educational spaces:
1. **Computer Laboratory:** High daytime load (40 PCs, dual AC, server rack).
2. **Classroom A:** Multimedia lecture hall (projector, smartboard, 12 tube fixtures).
3. **Classroom B:** Seminar room (ceiling fans, projector).
4. **Library Reading Room:** Extended evening study hours (OPAC terminals, study desk lamps).
5. **Faculty Office Suite:** Standard daytime office hours (desktops, mini-fridge).

### 6. 💵 Transparent Electricity Bill Estimator
- Itemizes volumetric energy charges, fixed monthly connection fees, peak demand charges ($\$/\text{kW}$), and local municipal tax surcharges.
- Compares current billing period with previous cycles to highlight consumption trends.

### 7. 📑 Telemetry Export & Reporting
- Download filtered consumption telemetry (CSV).
- Download detected anomaly audit logs (CSV).
- Download prioritized intelligent recommendations (CSV).
- Export an Executive Energy Audit Briefing report in Markdown.

---

## 🏗️ Architecture & Technology Stack

```
WattWise-AI/
│
├── app.py                     # Main Streamlit interactive dashboard
├── requirements.txt           # Python dependency specifications
├── README.md                  # Project overview and run guide
├── .gitignore                 # Version control exclusions
│
├── data/
│   ├── sample_energy_data.csv # 180-day hourly synthetic dataset (21,600 rows)
│   └── wattwise.db            # SQLite persistent database
│
├── models/
│   ├── forecaster_*.joblib    # Serialized RandomForest forecasting models
│   └── anomaly_detector_*.joblib # Serialized IsolationForest models
│
├── src/
│   ├── __init__.py
│   ├── database.py            # SQLite schema, queries, and audit logging
│   ├── data_generator.py      # Reproducible 180-day synthetic telemetry generator
│   ├── forecasting.py         # RandomForestRegressor & HourlyBaseline pipelines
│   ├── anomaly_detection.py   # IsolationForest anomaly scoring
│   ├── waste_intelligence.py  # Flagship Waste Intelligence Engine
│   ├── waste_detective.py     # Energy Waste Detective & forensic scenario engine
│   ├── recommendations.py     # Deterministic rule-based advisory engine
│   ├── savings_calculator.py  # Physics-based What-If simulator formulas
│   ├── bill_estimator.py      # Itemized electricity bill calculator
│   └── reporting.py           # CSV exporters and executive report generator
│
├── tests/
│   ├── __init__.py
│   ├── test_savings_calculator.py # Tests for kWh, savings, and ROI formulas
│   ├── test_bill_estimator.py     # Tests for tariffs, demand charges, and taxes
│   ├── test_forecasting.py        # Tests for features, baseline model, and RF
│   ├── test_anomaly_detection.py  # Tests for IsolationForest and waste diagnosis
│   ├── test_waste_detective.py    # Tests for calculations, forensic pipeline, & SQLite persistence
│   └── test_database.py           # Tests for SQLite tables, idempotency, and reset
│
├── scripts/
│   ├── generate_data.py       # CLI script to generate data and seed SQLite
│   └── train_models.py        # CLI script to train and serialize all models
│
└── docs/
    └── PROJECT_EXPLANATION.md # Comprehensive guide, architecture, judge Q&A
```

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Responsive, high-performance web dashboard |
| **Data Processing** | Pandas, NumPy | Telemetry aggregation, feature engineering |
| **Machine Learning** | Scikit-Learn | RandomForestRegressor & IsolationForest |
| **Visualization** | Plotly | Interactive time-series charts, heatmaps, and pies |
| **Model Storage** | Joblib | Serialization and fast loading of trained ML pipelines |
| **Database** | SQLite3 | Local zero-configuration relational storage |
| **Test Suite** | Pytest | Automated unit and integration testing |

---

## 🚀 Quickstart Guide (Windows PowerShell)

### Prerequisites
- Python 3.10 or Python 3.11 installed on your system.
- VS Code or your preferred code editor.

### 1. Clone or Open Workspace
```powershell
cd C:\Users\DELL\Desktop\WattWise-Ai
```

### 2. Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Generate Synthetic Dataset & Initialize Database
```powershell
python scripts\generate_data.py --days 180
```
*This generates 21,600 hourly records across 5 locations, saves `data/sample_energy_data.csv`, and seeds `data/wattwise.db`.*

### 5. Train Machine Learning Models
```powershell
python scripts\train_models.py
```
*Trains and evaluates RandomForest forecasting models and IsolationForest detectors for all 5 monitored rooms, saving the trained artifacts into `models/`.*

### 6. Run Automated Tests
```powershell
pytest -v
```
*Executes all 30 automated unit tests across data structures, physics formulas, forensic detective logic, and ML pipelines.*

### 7. Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
*Open your browser and navigate to `http://localhost:8501`.*

---

## 🎪 How to Demonstrate the Project to Judges

1. **Executive Overview (Tab 1):**  
   Point out the high-level KPIs: Total Consumed (kWh), Estimated Cost ($), and Waste Avoidance Potential. Highlight that all metrics reflect realistic continuous hourly telemetry across 5 campus locations.
2. **Live Demo Mode (Tab 2):**  
   - Show the live telemetry feed for the **Computer Laboratory**.
   - Click **"🚨 Trigger Demo Anomaly"**.
   - Watch the red alert banner immediately trigger with measured vs expected consumption, cost leakage, root cause diagnosis, and prescribed actions.
   - Show the audit log table at the bottom of the page.
   - Click **"Reset Demo State"** to restore clean baseline telemetry.
3. **Energy Waste Intelligence Engine (Tab 3):**  
   - Demonstrate the **Diagnostic Inspector**: slide the time to 21:00 (off-hours), set occupancy to 0, and energy draw to 4.8 kWh. Explain how it identifies **"CONFIRMED ELECTRICITY WASTE"** because computers/AC were left running without occupants.
   - Now slide occupancy to 35: observe how the engine changes classification to **"ELEVATED OPERATIONAL DEMAND (LEGITIMATE LOAD)"**, proving the AI distinguishes true waste from busy lab sessions!
4. **Energy Waste Detective (Tab 4):**  
   - Show the **Forensic Event Timeline** mapping past excess consumption events.
   - Expand the **Closed-Hours Surge Simulation** panel. Click **"Run Scenario Through Real Detection Pipeline"** to simulate an unexpected 5.4 kWh spike at 10:30 PM.
   - Show the resulting **Forensic Finding Panel**:
     - Baseline vs. Observed draw ($\Delta\text{kWh}$ and $\$\text{ cost}$).
     - Evidence confidence indicator (e.g. High Evidence 92%).
     - Plausible contributing factors (hypotheses under investigation).
     - Recommended practical action.
   - Update the investigation status from `PENDING` to `CONFIRMED_WASTE`, write auditor notes, and click **"Save Investigation Status to SQLite"**. Show that it persists across app reloads.
   - Click **"Export Detective Findings (CSV)"** to demonstrate the CSV audit trail.
5. **AI Electricity Demand Forecasting (Tab 5):**  
   - Show the held-out test evaluation metrics (Random Forest MAE/RMSE vs Baseline Model).
   - Show the **Forward Next-24-Hour Horizon Forecast** with confidence intervals.
6. **Energy Savings Lab (Tab 6):**  
   - Load the "Standard Desktops vs Eco Mini-PCs" preset.
   - Show the immediate reduction in daily kWh, annual financial savings, and simple payback period.
7. **Reports and Settings (Tab 9):**  
   - Show 1-click CSV telemetry and anomaly export.
   - Show the auto-generated Executive Audit Report in Markdown.

---

## ⚠️ Transparent Limitations & Disclaimers
1. **Synthetic Telemetry:** All data is synthetically generated via deterministic physics formulas with diurnal and seasonal cycles. It is designed for demonstration and simulation without physical smart meters.
2. **Offline Local Execution:** The core application runs completely offline on Windows without requiring paid LLM API keys or cloud subscriptions.
3. **Utility Billing Approximations:** Utility bill calculations use user-configured volumetric rates and fixed charges. Actual municipal tariffs may include complex tiered step-slabs or power-factor penalties.

---

## 🔮 Future Enhancements
- **Hardware Integration:** Connect Modbus / MQTT smart energy meters and CT clamp sensors for real-time live laboratory telemetry.
- **Automated Relay Actuation:** Integrate IoT relays (e.g. Sonoff, Shelly) to automatically disconnect idle circuits upon detected waste.
- **Local Small LLM Copilot:** Integrate local Ollama / LLaMA-3 models for natural-language conversational queries regarding facility consumption.

---
*Created with ❤️ for energy conservation and sustainable campus computing.*

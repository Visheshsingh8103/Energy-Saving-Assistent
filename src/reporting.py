"""
src/reporting.py
Reporting and Data Export module for WattWise AI.
Generates structured CSV exports, audit summaries, and executive briefing reports.
"""

from typing import Dict, Any, List, Optional
import io
import pandas as pd


def export_consumption_csv(df: pd.DataFrame) -> str:
    """Generate CSV string of filtered consumption telemetry."""
    if df.empty:
        return "timestamp,location_id,location_name,hour,occupancy,temperature,consumption_kwh,is_anomaly_flag\n"
    
    cols = [
        c for c in [
            "timestamp", "location_id", "location_name", "hour", "day_of_week",
            "occupancy", "temperature", "consumption_kwh", "is_anomaly_flag",
            "anomaly_score", "is_demo_injected"
        ] if c in df.columns
    ]
    return df[cols].to_csv(index=False)


def export_anomalies_csv(df: pd.DataFrame) -> str:
    """Generate CSV string of detected anomalous events."""
    if df.empty or "is_anomaly_flag" not in df.columns:
        return "timestamp,location_name,measured_kwh,expected_kwh,diff_kwh,cost_diff,suggested_investigation\n"

    anom_df = df[df["is_anomaly_flag"] == 1].copy()
    if anom_df.empty:
        return "timestamp,location_name,measured_kwh,expected_kwh,diff_kwh,cost_diff,suggested_investigation\n"

    cols = [
        c for c in [
            "timestamp", "location_name", "consumption_kwh", "expected_kwh",
            "diff_kwh", "cost_diff", "occupancy", "temperature", "anomaly_score",
            "suggested_investigation"
        ] if c in anom_df.columns
    ]
    return anom_df[cols].to_csv(index=False)


def export_recommendations_csv(recommendations: List[Dict[str, Any]]) -> str:
    """Generate CSV string of prioritized energy recommendations."""
    if not recommendations:
        return "title,location_name,confidence_level,estimated_savings_kwh,estimated_savings_cost,reason,suggested_action\n"
    
    df = pd.DataFrame(recommendations)
    cols = [
        c for c in [
            "title", "location_name", "confidence_level",
            "estimated_savings_kwh", "estimated_savings_cost",
            "reason", "supporting_data", "suggested_action"
        ] if c in df.columns
    ]
    return df[cols].to_csv(index=False)


def generate_executive_summary_markdown(
    location_name: str,
    date_range_str: str,
    total_kwh: float,
    total_cost: float,
    peak_demand_kw: float,
    anomalies_count: int,
    waste_kwh: float,
    waste_cost: float,
    forecast_next_24h_kwh: Optional[float] = None,
    actionable_recs_count: int = 0,
    currency_symbol: str = "$"
) -> str:
    """
    Format an executive markdown report suitable for facility managers and audit committees.
    """
    report = f"""# ⚡ WATTWISE AI — EXECUTIVE ENERGY AUDIT REPORT
**Facility / Location:** {location_name}  
**Audited Period:** {date_range_str}  
**Generated On:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Mode:** DEMO SIMULATION (Synthetically Verified Telemetry)

---

### 1. High-Level Consumption & Cost Summary
| Metric | Value | Reference / Context |
| :--- | :--- | :--- |
| **Total Energy Consumed** | **{total_kwh:,.2f} kWh** | Cumulative recorded draw |
| **Total Estimated Utility Cost** | **{currency_symbol}{total_cost:,.2f}** | Based on active tariff profile |
| **Peak Demand Recorded** | **{peak_demand_kw:.2f} kW** | Max concurrent hourly draw |
| **Average Hourly Intensity** | **{(total_kwh / max(1, 24 * 30)):.2f} kWh/hr** | Baseline normalized demand |

---

### 2. Waste Intelligence & Anomaly Audit
- **Flagged Statistical Anomalies:** `{anomalies_count} events`
- **Diagnosed Idle / Off-Hours Waste:** `{waste_kwh:,.2f} kWh`
- **Avoidable Financial Leakage:** `{currency_symbol}{waste_cost:,.2f}`
- **Waste Ratio:** `{(waste_kwh / max(0.01, total_kwh)) * 100:.1f}%` of total power consumed

> **Key Diagnostic Finding:**  
> The Energy Waste Intelligence Engine detected recurring off-hours loads during vacant periods. Implementing automated load shed or sub-circuit isolation could eliminate the majority of avoidable expenditure.

---

### 3. Predictive Horizon & Recommendations
"""
    if forecast_next_24h_kwh is not None:
        report += f"- **Projected Next-24-Hour Demand:** `{forecast_next_24h_kwh:.2f} kWh` (Estimated Cost: `{currency_symbol}{forecast_next_24h_kwh * (total_cost / max(0.1, total_kwh)):.2f}`)\n"
    
    report += f"""- **Active Energy Optimization Actions:** `{actionable_recs_count} prioritized recommendations ready for deployment`

---
*Report generated automatically by WattWise AI — Detect Waste. Predict Demand. Save Energy.*
"""
    return report

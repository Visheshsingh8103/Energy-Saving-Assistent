"""
src/waste_intelligence.py
Energy Waste Intelligence Engine for WattWise AI.
The flagship analytics engine combining historical baselines, calendar context,
real-time occupancy, operating schedules, and machine learning anomaly scores
to identify, categorize, quantify, and explain suspicious energy consumption.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class EnergyWasteIntelligenceEngine:
    """
    Dedicated intelligence engine that synthesizes physical room models,
    operational schedules, occupancy states, and ML anomaly indicators
    to distinguish genuine energy waste from legitimate high operational load.
    """
    def __init__(
        self,
        operating_start: int = 8,
        operating_end: int = 18,
        tariff_per_kwh: float = 0.15,
        waste_threshold_kwh: float = 0.5
    ):
        self.operating_start = operating_start
        self.operating_end = operating_end
        self.tariff_per_kwh = tariff_per_kwh
        self.waste_threshold_kwh = waste_threshold_kwh

    def analyze_event(
        self,
        location_id: str,
        location_name: str,
        timestamp: str,
        consumption_kwh: float,
        expected_kwh: float,
        occupancy: int,
        temperature: float,
        is_anomaly: bool = False,
        anomaly_score: float = 0.0,
        baseline_closed_kwh: float = 0.5,
        baseline_open_kwh: float = 2.0
    ) -> Dict[str, Any]:
        """
        Analyze a single reading in context and generate a complete diagnostic explanation.
        """
        dt = pd.to_datetime(timestamp)
        hour = dt.hour
        dow = dt.weekday()
        is_weekend = (dow >= 5)

        is_operating_hours = (self.operating_start <= hour < self.operating_end) and not is_weekend
        diff_kwh = round(consumption_kwh - expected_kwh, 3)
        cost_impact = round(diff_kwh * self.tariff_per_kwh, 3)

        # Waste classification logic
        status = "NORMAL OPERATION"
        category = "OPTIMAL_LOAD"
        severity = "INFO"
        confidence_pct = 95
        possible_explanations = []
        recommended_actions = []

        if diff_kwh > self.waste_threshold_kwh or is_anomaly:
            if not is_operating_hours and occupancy == 0 and consumption_kwh > (baseline_closed_kwh * 1.6):
                status = "CONFIRMED ELECTRICITY WASTE"
                category = "OFF_HOURS_IDLE_WASTE"
                severity = "HIGH" if diff_kwh > 2.0 else "MEDIUM"
                confidence_pct = 92
                possible_explanations = [
                    f"Room is outside operating hours ({self.operating_start}:00 - {self.operating_end}:00) with 0 occupants.",
                    f"Consumption of {consumption_kwh:.2f} kWh is significantly above closed baseline of {baseline_closed_kwh:.2f} kWh.",
                    "Air conditioning units, projector, or computer lab workstations were left powered on overnight."
                ]
                recommended_actions = [
                    "Dispatch night security or custodian to verify room power switches.",
                    "Implement centralized automated shutdown schedule at closing hour.",
                    "Check AC remote controls to ensure sleep timer is active."
                ]

            elif is_operating_hours and occupancy == 0 and consumption_kwh > (baseline_open_kwh * 0.9):
                status = "SUSPECTED ELECTRICITY WASTE"
                category = "UNOCCUPIED_DAYTIME_WASTE"
                severity = "MEDIUM"
                confidence_pct = 85
                possible_explanations = [
                    "Room is currently unoccupied (0 students / personnel) during scheduled hours.",
                    f"High continuous draw of {consumption_kwh:.2f} kWh suggests full lighting and AC running in an empty room.",
                    "Class may have concluded early without turning off lighting or audiovisual gear."
                ]
                recommended_actions = [
                    "Install PIR passive infrared motion sensors to auto-dim lighting after 15 minutes of inactivity.",
                    "Encourage faculty and lab assistants to switch off multimedia projectors when leaving."
                ]

            elif occupancy > 10 and diff_kwh > 0:
                status = "ELEVATED OPERATIONAL DEMAND (LEGITIMATE LOAD)"
                category = "HIGH_CONCURRENT_USE"
                severity = "LOW"
                confidence_pct = 88
                possible_explanations = [
                    f"Elevated consumption ({consumption_kwh:.2f} kWh) correlates directly with high occupancy ({occupancy} active users).",
                    "High workload, lab practical session, or project demonstration in progress.",
                    "This is an operational peak rather than wasted energy."
                ]
                recommended_actions = [
                    "No immediate intervention required. Ensure equipment is powered down once session terminates.",
                    "Review time-of-use tariff to shift heavy computation to off-peak slots if possible."
                ]

            elif temperature > 32.0 and diff_kwh > 0:
                status = "WEATHER-DRIVEN THERMAL LOAD"
                category = "COOLING_DEMAND"
                severity = "LOW"
                confidence_pct = 82
                possible_explanations = [
                    f"Ambient outdoor temperature reached {temperature:.1f}°C, increasing HVAC thermal load.",
                    "Compressor duty cycle is running at maximum to maintain indoor setpoint temperature."
                ]
                recommended_actions = [
                    "Ensure windows, blinds, and doors remain closed to prevent thermal ingress.",
                    "Adjust thermostat setting upward by 1-2°C (e.g. from 21°C to 24°C) for up to 10% HVAC energy savings."
                ]

            elif diff_kwh < -0.4:
                status = "UNUSUALLY LOW CONSUMPTION"
                category = "UNDER_CONSUMPTION"
                severity = "LOW"
                confidence_pct = 75
                possible_explanations = [
                    f"Measured energy ({consumption_kwh:.2f} kWh) is below expected floor ({expected_kwh:.2f} kWh).",
                    "Possible scheduled holiday, class cancellation, or tripped circuit breaker."
                ]
                recommended_actions = [
                    "Verify if room was scheduled for maintenance or if power feed was tripped."
                ]

            else:
                status = "STATISTICAL DEVIATION"
                category = "MARGINAL_VARIANCE"
                severity = "LOW"
                confidence_pct = 70
                possible_explanations = [
                    f"Observed draw of {consumption_kwh:.2f} kWh is slightly above historical expectation ({expected_kwh:.2f} kWh).",
                    "Marginal fluctuation within acceptable operational tolerance."
                ]
                recommended_actions = [
                    "Continue monitoring; no immediate manual action warranted."
                ]
        else:
            possible_explanations = [
                f"Energy consumption ({consumption_kwh:.2f} kWh) is congruent with historical average ({expected_kwh:.2f} kWh).",
                "Operating schedule, occupancy levels, and equipment load are balanced."
            ]
            recommended_actions = [
                "Continue standard energy stewardship protocols."
            ]

        return {
            "timestamp": timestamp,
            "location_id": location_id,
            "location_name": location_name,
            "hour": hour,
            "occupancy": occupancy,
            "temperature": temperature,
            "measured_kwh": round(consumption_kwh, 3),
            "expected_kwh": round(expected_kwh, 3),
            "diff_kwh": diff_kwh,
            "estimated_extra_cost": max(0.0, cost_impact),
            "status": status,
            "category": category,
            "severity": severity,
            "confidence_pct": confidence_pct,
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": round(anomaly_score, 3),
            "possible_explanations": possible_explanations,
            "recommended_actions": recommended_actions,
            "is_waste": status.startswith("CONFIRMED") or status.startswith("SUSPECTED")
        }

    def audit_dataset_for_waste(
        self,
        df: pd.DataFrame,
        locations_meta: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Scan an entire historical dataframe, compute expected hourly baselines,
        and generate an executive Waste Audit report.
        """
        if df.empty:
            return {"total_waste_kwh": 0.0, "total_waste_cost": 0.0, "waste_events_count": 0, "events": []}

        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy["timestamp"]):
            df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

        df_copy["hour"] = df_copy["timestamp"].dt.hour
        df_copy["day_of_week"] = df_copy["timestamp"].dt.dayofweek
        df_copy["is_weekend"] = (df_copy["day_of_week"] >= 5).astype(int)

        # Baseline per location and hour
        baselines = df_copy.groupby(["location_id", "hour", "is_weekend"])["consumption_kwh"].median().to_dict()

        waste_events = []
        total_waste_kwh = 0.0

        for _, row in df_copy.iterrows():
            loc_id = row["location_id"]
            hr = row["hour"]
            is_wknd = row["is_weekend"]
            expected = baselines.get((loc_id, hr, is_wknd), float(row["consumption_kwh"]))

            meta = (locations_meta or {}).get(loc_id, {})
            base_closed = meta.get("baseline_kw_closed", 0.5)
            base_open = meta.get("baseline_kw_open", 2.0)

            diag = self.analyze_event(
                location_id=loc_id,
                location_name=row.get("location_name", loc_id),
                timestamp=str(row["timestamp"]),
                consumption_kwh=float(row["consumption_kwh"]),
                expected_kwh=float(expected),
                occupancy=int(row.get("occupancy", 0)),
                temperature=float(row.get("temperature", 24.0)),
                is_anomaly=bool(row.get("is_anomaly_flag", 0)),
                anomaly_score=float(row.get("anomaly_score", 0.0)),
                baseline_closed_kwh=base_closed,
                baseline_open_kwh=base_open
            )

            if diag["is_waste"]:
                waste_events.append(diag)
                total_waste_kwh += max(0.0, diag["diff_kwh"])

        total_waste_cost = round(total_waste_kwh * self.tariff_per_kwh, 2)

        return {
            "total_waste_kwh": round(total_waste_kwh, 2),
            "total_waste_cost": total_waste_cost,
            "waste_events_count": len(waste_events),
            "waste_events": waste_events[-100:] # Return recent 100 events
        }

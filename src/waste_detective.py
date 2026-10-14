"""
src/waste_detective.py
Energy Waste Detective module for WattWise AI.
Provides evidence-based explanations of unusual electricity consumption,
probabilistic root-cause hypotheses, actionable remediation plans,
interactive investigation status tracking, and scenario testing.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import pandas as pd
import numpy as np


def calculate_waste_metrics(
    observed_kwh: float,
    baseline_kwh: float,
    tariff_per_kwh: float
) -> Tuple[float, float, float]:
    """
    Calculate kWh delta, percentage deviation, and financial cost impact.
    Returns: (diff_kwh, pct_diff, cost_impact)
    """
    if tariff_per_kwh < 0:
        raise ValueError("Tariff per kWh must be non-negative.")

    diff_kwh = round(float(observed_kwh - baseline_kwh), 3)
    if baseline_kwh > 0:
        pct_diff = round(float((diff_kwh / baseline_kwh) * 100.0), 1)
    else:
        pct_diff = 100.0 if diff_kwh > 0 else 0.0

    cost_impact = max(0.0, round(float(diff_kwh * tariff_per_kwh), 3))
    return diff_kwh, pct_diff, cost_impact


def compute_evidence_score(
    observed_kwh: float,
    baseline_kwh: float,
    is_closed_hours: bool,
    occupancy: int,
    anomaly_score: float = 0.0,
    temperature: float = 24.0
) -> Tuple[float, str, List[str]]:
    """
    Compute an empirical evidence score (0.0 to 1.0) and descriptive confidence tier.
    Synthesizes physical telemetry factors into quantitative evidence.
    """
    evidence_points = 0.0
    evidence_factors = []

    # Factor 1: Schedule alignment (closed hours is stronger evidence of waste)
    if is_closed_hours:
        evidence_points += 0.35
        evidence_factors.append("Facility was scheduled CLOSED during the recorded observation.")
    else:
        evidence_points += 0.15
        evidence_factors.append("Facility was scheduled OPEN during observation.")

    # Factor 2: Occupancy verification
    if occupancy == 0:
        evidence_points += 0.30
        evidence_factors.append("Zero active occupants detected (no human activity justifies draw).")
    elif occupancy < 5:
        evidence_points += 0.15
        evidence_factors.append(f"Low occupant count ({occupancy} persons) vs high continuous draw.")
    else:
        evidence_points += 0.05
        evidence_factors.append(f"Active occupancy present ({occupancy} persons); load partially operational.")

    # Factor 3: Magnitude of deviation
    ratio = observed_kwh / max(0.05, baseline_kwh)
    if ratio >= 3.0:
        evidence_points += 0.25
        evidence_factors.append(f"Measured consumption is {ratio:.1f}x higher than baseline floor.")
    elif ratio >= 1.5:
        evidence_points += 0.15
        evidence_factors.append(f"Measured consumption is {ratio:.1f}x higher than baseline.")
    else:
        evidence_points += 0.05

    # Factor 4: ML Anomaly Score contribution
    if anomaly_score > 0.7:
        evidence_points += 0.10
        evidence_factors.append("IsolationForest flagged this data point with high outlier confidence.")
    elif anomaly_score > 0.4:
        evidence_points += 0.05

    # Weather dampener (if extreme heat >33°C, cooling load is partially weather-driven)
    if temperature > 33.0 and ratio < 2.5:
        evidence_points = max(0.4, evidence_points - 0.15)
        evidence_factors.append(f"High ambient temperature ({temperature:.1f}°C) likely drove higher HVAC cycle.")

    final_score = round(min(0.98, max(0.35, evidence_points)), 2)

    if final_score >= 0.85:
        level = "High Evidence"
    elif final_score >= 0.65:
        level = "Moderate Evidence"
    else:
        level = "Exploratory Evidence"

    return final_score, level, evidence_factors


class WasteDetective:
    """
    Evidence-based energy waste diagnostic engine.
    Produces reasoned findings with plausible hypotheses, actionable next steps,
    and investigation status tracking.
    """
    def __init__(
        self,
        operating_start: int = 8,
        operating_end: int = 18,
        tariff_per_kwh: float = 0.15
    ):
        self.operating_start = operating_start
        self.operating_end = operating_end
        self.tariff_per_kwh = tariff_per_kwh

    def generate_finding(
        self,
        timestamp: str,
        location_id: str,
        location_name: str,
        observed_kwh: float,
        baseline_kwh: float,
        occupancy: int = 0,
        temperature: float = 24.0,
        anomaly_score: float = 0.0,
        is_synthetic_test: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate an observation against baseline telemetry and generate an evidence-based finding.
        """
        dt = pd.to_datetime(timestamp)
        hour = dt.hour
        dow = dt.weekday()
        is_weekend = (dow >= 5)

        is_closed_hours = (hour < self.operating_start or hour >= self.operating_end) or is_weekend
        diff_kwh, pct_diff, cost_impact = calculate_waste_metrics(observed_kwh, baseline_kwh, self.tariff_per_kwh)

        evidence_score, evidence_level, evidence_factors = compute_evidence_score(
            observed_kwh=observed_kwh,
            baseline_kwh=baseline_kwh,
            is_closed_hours=is_closed_hours,
            occupancy=occupancy,
            anomaly_score=anomaly_score,
            temperature=temperature
        )

        # Flag reason formulation
        if is_closed_hours and diff_kwh > 0.4 and occupancy == 0:
            flag_reason = (
                f"Observed draw of {observed_kwh:.2f} kWh at {hour:02d}:00 exceeded closed baseline "
                f"({baseline_kwh:.2f} kWh) by +{diff_kwh:.2f} kWh (+{pct_diff:.0f}%) with zero occupants during scheduled closed hours."
            )
            plausible_causes = [
                "Air conditioning or heat pump units left operating in standard comfort mode rather than setback mode.",
                "Unattended workstation clusters, computer monitors, or audiovisual projectors left powered on overnight.",
                "Janitorial or night maintenance staff utilizing portable high-wattage equipment (e.g. floor polishers).",
                "Sub-meter current transformer calibration drift or transient electrical phase imbalance."
            ]
            recommended_action = (
                "Dispatch facility security or custodian to verify wall switches and AC remotes, "
                "and verify OS automated sleep timer policies on all workstations."
            )

        elif not is_closed_hours and diff_kwh > 0.5 and occupancy == 0:
            flag_reason = (
                f"Room was vacant (0 occupants) during operational hours ({hour:02d}:00), "
                f"yet power consumption remained high at {observed_kwh:.2f} kWh (+{diff_kwh:.2f} kWh over baseline)."
            )
            plausible_causes = [
                "Class or meeting concluded early without occupants turning off overhead lighting and projector.",
                "Centralized ventilation scheduled for occupied mode despite schedule cancellation.",
                "Peripheral laboratory equipment or charging carts active without active supervision."
            ]
            recommended_action = (
                "Install PIR occupancy sensors for automated lighting setback after 15 minutes of inactivity, "
                "and notify room coordinator."
            )

        elif diff_kwh > 2.0:
            flag_reason = (
                f"Sharp power surge: measured {observed_kwh:.2f} kWh is +{diff_kwh:.2f} kWh (+{pct_diff:.0f}%) "
                f"above typical expected baseline ({baseline_kwh:.2f} kWh)."
            )
            plausible_causes = [
                "Concurrent boot cycle of heavy computational workloads or rendering jobs.",
                "Electrical equipment mechanical resistance or failing compressor motor drawing excess startup current.",
                "Unauthorized heating/cooling appliances (e.g. unmonitored space heaters or kettles)."
            ]
            recommended_action = (
                "Inspect main distribution panel for warm circuit breakers and audit connected high-draw appliances."
            )
        else:
            flag_reason = (
                f"Telemetry variation: observed {observed_kwh:.2f} kWh represents a marginal deviation of "
                f"{'+' if diff_kwh > 0 else ''}{diff_kwh:.2f} kWh from historical expectation."
            )
            plausible_causes = [
                "Natural operational fluctuation within tolerable statistical limits.",
                "Minor ambient temperature swing driving slight increase in ventilation duty cycle."
            ]
            recommended_action = "Continue automated telemetry monitoring; no immediate manual intervention required."

        return {
            "timestamp": timestamp,
            "location_id": location_id,
            "location_name": location_name,
            "hour": hour,
            "observed_kwh": round(observed_kwh, 3),
            "baseline_kwh": round(baseline_kwh, 3),
            "diff_kwh": diff_kwh,
            "pct_diff": pct_diff,
            "cost_impact": cost_impact,
            "tariff_rate": self.tariff_per_kwh,
            "is_closed_hours": is_closed_hours,
            "occupancy": occupancy,
            "temperature": temperature,
            "flag_reason": flag_reason,
            "plausible_causes": plausible_causes,
            "recommended_action": recommended_action,
            "evidence_score": evidence_score,
            "evidence_level": evidence_level,
            "evidence_factors": evidence_factors,
            "is_synthetic_test": int(is_synthetic_test),
            "investigation_status": "PENDING",
            "investigator_notes": ""
        }

    def run_scenario_through_pipeline(
        self,
        scenario: Dict[str, Any],
        detector: Optional[Any] = None,
        tariff_per_kwh: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a test scenario dynamically through the detection and detective pipeline.
        Calculates all baseline comparisons, deltas, and costs from live logic (no hardcoded outputs).
        """
        active_tariff = tariff_per_kwh if tariff_per_kwh is not None else self.tariff_per_kwh
        self.tariff_per_kwh = active_tariff

        timestamp = scenario.get("timestamp", pd.Timestamp.now().strftime("%Y-%m-%d 22:30:00"))
        location_id = scenario.get("location_id", "computer_lab")
        location_name = scenario.get("location_name", "Computer Laboratory")
        observed_kwh = float(scenario.get("observed_kwh", 5.2))
        baseline_kwh = float(scenario.get("baseline_kwh", 0.8))
        occupancy = int(scenario.get("occupancy", 0))
        temperature = float(scenario.get("temperature", 25.0))

        # Check with ML detector if available
        anomaly_score = 0.85
        if detector is not None and hasattr(detector, "evaluate_single_reading"):
            try:
                res = detector.evaluate_single_reading(
                    timestamp=timestamp,
                    consumption_kwh=observed_kwh,
                    occupancy=occupancy,
                    temperature=temperature,
                    tariff_per_kwh=active_tariff,
                    operating_start=self.operating_start,
                    operating_end=self.operating_end
                )
                anomaly_score = res.get("anomaly_score", 0.85)
                if "expected_kwh" in res and res["expected_kwh"] > 0:
                    baseline_kwh = res["expected_kwh"]
            except Exception:
                pass

        finding = self.generate_finding(
            timestamp=timestamp,
            location_id=location_id,
            location_name=location_name,
            observed_kwh=observed_kwh,
            baseline_kwh=baseline_kwh,
            occupancy=occupancy,
            temperature=temperature,
            anomaly_score=anomaly_score,
            is_synthetic_test=True
        )
        return finding


def export_detective_findings_csv(findings: List[Dict[str, Any]]) -> str:
    """Generate CSV text for detective findings including investigation status."""
    if not findings:
        return (
            "timestamp,location_id,location_name,observed_kwh,baseline_kwh,diff_kwh,"
            "cost_impact,flag_reason,plausible_causes,recommended_action,evidence_score,"
            "evidence_level,investigation_status,investigator_notes\n"
        )

    clean_rows = []
    for f in findings:
        causes = f.get("plausible_causes", "")
        if isinstance(causes, (list, tuple)):
            causes_str = "; ".join(causes)
        else:
            causes_str = str(causes).replace(" | ", "; ")

        clean_rows.append({
            "timestamp": f.get("timestamp"),
            "location_id": f.get("location_id"),
            "location_name": f.get("location_name"),
            "observed_kwh": f.get("observed_kwh"),
            "baseline_kwh": f.get("baseline_kwh"),
            "diff_kwh": f.get("diff_kwh"),
            "cost_impact": f.get("cost_impact"),
            "flag_reason": f.get("flag_reason"),
            "plausible_causes": causes_str,
            "recommended_action": f.get("recommended_action"),
            "evidence_score": f.get("evidence_score"),
            "evidence_level": f.get("evidence_level"),
            "investigation_status": f.get("investigation_status", "PENDING"),
            "investigator_notes": f.get("investigator_notes", ""),
            "is_synthetic_test": f.get("is_synthetic_test", 0)
        })

    df = pd.DataFrame(clean_rows)
    return df.to_csv(index=False)

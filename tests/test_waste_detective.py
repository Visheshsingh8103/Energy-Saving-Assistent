"""
tests/test_waste_detective.py
Unit tests for the Energy Waste Detective module:
- Mathematical calculations (delta kWh, cost with configured tariff)
- Evidence scoring and causal explanation logic
- Detection pipeline execution for closed-hours scenarios
- Investigation status persistence in SQLite database
- CSV report generation
"""

import os
import tempfile
import pytest
import pandas as pd
from src.database import (
    init_db,
    save_detective_finding,
    get_detective_findings,
    update_investigation_status,
    reset_demo_data
)
from src.waste_detective import (
    calculate_waste_metrics,
    compute_evidence_score,
    WasteDetective,
    export_detective_findings_csv
)


# ==============================================================================
# 1. CALCULATION TESTS
# ==============================================================================
def test_calculate_waste_metrics_standard():
    # Observed: 5.0 kWh, Baseline: 1.0 kWh, Tariff: $0.20/kWh
    # Delta = 4.0 kWh, Pct = +400%, Cost = 4.0 * 0.20 = $0.80
    diff_kwh, pct_diff, cost = calculate_waste_metrics(
        observed_kwh=5.0,
        baseline_kwh=1.0,
        tariff_per_kwh=0.20
    )
    assert diff_kwh == 4.0
    assert pct_diff == 400.0
    assert cost == 0.80


def test_calculate_waste_metrics_under_consumption():
    # Observed: 0.5 kWh, Baseline: 1.0 kWh -> Delta = -0.5 kWh, Cost = 0.0 (no extra cost)
    diff_kwh, pct_diff, cost = calculate_waste_metrics(
        observed_kwh=0.5,
        baseline_kwh=1.0,
        tariff_per_kwh=0.15
    )
    assert diff_kwh == -0.5
    assert pct_diff == -50.0
    assert cost == 0.0


def test_calculate_waste_metrics_negative_tariff_raises():
    with pytest.raises(ValueError):
        calculate_waste_metrics(observed_kwh=5.0, baseline_kwh=1.0, tariff_per_kwh=-0.15)


# ==============================================================================
# 2. EVIDENCE SCORING & EXPLANATION PIPELINE TESTS
# ==============================================================================
def test_compute_evidence_score_closed_hours_unoccupied():
    # High evidence: Closed hours, zero occupants, massive 6x draw
    score, level, factors = compute_evidence_score(
        observed_kwh=6.0,
        baseline_kwh=1.0,
        is_closed_hours=True,
        occupancy=0,
        anomaly_score=0.85,
        temperature=24.0
    )
    assert score >= 0.85
    assert level == "High Evidence"
    assert any("CLOSED" in f for f in factors)
    assert any("Zero active occupants" in f for f in factors)


def test_detective_generates_plausible_causes_without_certainty():
    detective = WasteDetective(operating_start=8, operating_end=18, tariff_per_kwh=0.15)
    finding = detective.generate_finding(
        timestamp="2026-10-15 22:00:00",
        location_id="computer_lab",
        location_name="Computer Laboratory",
        observed_kwh=5.5,
        baseline_kwh=0.8,
        occupancy=0,
        temperature=23.0
    )

    # Check comparison
    assert finding["observed_kwh"] == 5.5
    assert finding["baseline_kwh"] == 0.8
    assert finding["diff_kwh"] == 4.7
    assert finding["cost_impact"] == round(4.7 * 0.15, 3)

    # Check reason and causes: must identify plausible causes without claiming certainty
    assert len(finding["plausible_causes"]) >= 3
    # Check that causes describe potential factors (hypotheses)
    causes_text = " ".join(finding["plausible_causes"]).lower()
    assert "left" in causes_text or "equipment" in causes_text or "override" in causes_text

    # Check practical recommendation
    assert len(finding["recommended_action"]) > 10

    # Evidence score
    assert finding["evidence_score"] >= 0.85
    assert finding["evidence_level"] == "High Evidence"


def test_run_scenario_through_real_pipeline():
    detective = WasteDetective(operating_start=8, operating_end=18, tariff_per_kwh=0.18)
    
    # Custom closed-hours rise scenario
    scenario = {
        "timestamp": "2026-10-15 23:15:00",
        "location_id": "classroom_a",
        "location_name": "Classroom A",
        "observed_kwh": 3.85,
        "baseline_kwh": 0.15,
        "occupancy": 0,
        "temperature": 24.5
    }

    finding = detective.run_scenario_through_pipeline(scenario, tariff_per_kwh=0.18)

    # Values must be dynamically calculated, not hardcoded
    expected_diff = round(3.85 - 0.15, 3) # 3.7
    expected_cost = round(expected_diff * 0.18, 3) # 0.666

    assert finding["diff_kwh"] == expected_diff
    assert finding["cost_impact"] == expected_cost
    assert finding["is_synthetic_test"] == 1
    assert finding["is_closed_hours"] is True
    assert "Classroom A" in finding["location_name"]


# ==============================================================================
# 3. SQLITE PERSISTENCE & INVESTIGATION STATUS TESTS
# ==============================================================================
@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_investigation_status_persistence(temp_db):
    detective = WasteDetective(operating_start=8, operating_end=18, tariff_per_kwh=0.15)
    finding = detective.generate_finding(
        timestamp="2026-10-15 21:00:00",
        location_id="computer_lab",
        location_name="Computer Laboratory",
        observed_kwh=4.8,
        baseline_kwh=0.8,
        occupancy=0
    )

    # 1. Save finding to SQLite
    finding_id = save_detective_finding(temp_db, finding)
    assert finding_id > 0

    # 2. Retrieve findings
    findings = get_detective_findings(temp_db, location_id="computer_lab")
    assert len(findings) == 1
    stored = findings[0]
    assert stored["observed_kwh"] == 4.8
    assert stored["investigation_status"] == "PENDING"
    assert stored["investigator_notes"] == ""

    # 3. Mark as INVESTIGATED with auditor notes
    updated = update_investigation_status(
        temp_db,
        finding_id=stored["id"],
        new_status="CONFIRMED_WASTE",
        investigator_notes="Security verified AC and 20 PCs left on after evening lab."
    )
    assert updated is True

    # 4. Re-query from database to verify persistence
    updated_findings = get_detective_findings(temp_db, location_id="computer_lab")
    assert len(updated_findings) == 1
    reloaded = updated_findings[0]
    assert reloaded["investigation_status"] == "CONFIRMED_WASTE"
    assert "Security verified" in reloaded["investigator_notes"]
    assert reloaded["investigated_at"] is not None


def test_reset_demo_data_cleans_synthetic_test_findings(temp_db):
    # Save a normal finding
    f_real = {
        "timestamp": "2026-10-10 10:00:00",
        "location_id": "library",
        "location_name": "Library",
        "observed_kwh": 3.0,
        "baseline_kwh": 1.5,
        "diff_kwh": 1.5,
        "cost_impact": 0.225,
        "is_synthetic_test": 0
    }
    # Save a synthetic test scenario
    f_test = {
        "timestamp": "2026-10-15 22:00:00",
        "location_id": "computer_lab",
        "location_name": "Computer Laboratory",
        "observed_kwh": 5.0,
        "baseline_kwh": 0.8,
        "diff_kwh": 4.2,
        "cost_impact": 0.63,
        "is_synthetic_test": 1
    }
    save_detective_finding(temp_db, f_real)
    save_detective_finding(temp_db, f_test)

    assert len(get_detective_findings(temp_db)) == 2

    # Reset demo data: synthetic test should be removed, real record preserved
    reset_demo_data(temp_db)
    remaining = get_detective_findings(temp_db)
    assert len(remaining) == 1
    assert remaining[0]["location_id"] == "library"


def test_export_detective_findings_csv():
    findings = [
        {
            "timestamp": "2026-10-15 22:00:00",
            "location_id": "computer_lab",
            "location_name": "Computer Laboratory",
            "observed_kwh": 5.2,
            "baseline_kwh": 0.8,
            "diff_kwh": 4.4,
            "cost_impact": 0.66,
            "flag_reason": "High off-hours draw",
            "plausible_causes": ["AC left on", "Desktops active"],
            "recommended_action": "Check switches",
            "evidence_score": 0.92,
            "evidence_level": "High Evidence",
            "investigation_status": "INVESTIGATED",
            "investigator_notes": "All confirmed",
            "is_synthetic_test": 0
        }
    ]
    csv_str = export_detective_findings_csv(findings)
    assert "timestamp,location_id" in csv_str
    assert "5.2" in csv_str
    assert "INVESTIGATED" in csv_str
    assert "All confirmed" in csv_str

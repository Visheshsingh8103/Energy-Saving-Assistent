"""
tests/test_anomaly_detection.py
Unit tests for the AI Anomaly Detection pipeline and waste diagnostic logic.
"""

import os
import pytest
import pandas as pd
from src.data_generator import generate_synthetic_energy_dataset
from src.anomaly_detection import EnergyAnomalyDetector, extract_anomaly_features
from src.waste_intelligence import EnergyWasteIntelligenceEngine


@pytest.fixture(scope="module")
def sample_dataset():
    df = generate_synthetic_energy_dataset(days=180, location_ids=["computer_lab"], seed=42)
    return df


def test_extract_anomaly_features(sample_dataset):
    df_feat, hourly_means = extract_anomaly_features(sample_dataset)
    assert "hourly_mean_baseline" in df_feat.columns
    assert "diff_from_hourly_mean" in df_feat.columns
    assert len(hourly_means) == 24


def test_anomaly_detector_train_and_predict(sample_dataset, tmp_path):
    detector = EnergyAnomalyDetector(location_id="test_lab", contamination=0.05)
    test_model_path = os.path.join(tmp_path, "detector_test_lab.joblib")
    detector.get_model_path = lambda: test_model_path

    detector.fit(sample_dataset)
    assert detector.model is not None
    assert detector.total_trained_samples == len(sample_dataset)
    assert os.path.exists(test_model_path)

    # Predict
    test_subset = sample_dataset.iloc[-200:].copy()
    pred_df = detector.predict(test_subset, tariff_per_kwh=0.15)
    assert "is_anomaly_flag" in pred_df.columns
    assert "anomaly_score" in pred_df.columns
    assert "expected_kwh" in pred_df.columns
    assert "diff_kwh" in pred_df.columns
    assert "suggested_investigation" in pred_df.columns


def test_evaluate_single_reading_distinction(sample_dataset, tmp_path):
    detector = EnergyAnomalyDetector(location_id="test_lab", contamination=0.05)
    test_model_path = os.path.join(tmp_path, "detector_test_lab.joblib")
    detector.get_model_path = lambda: test_model_path
    detector.fit(sample_dataset)

    # 1. Normal daytime reading matching the expected hourly mean
    mean_14 = detector.hourly_means.get(14, 5.0)
    normal_res = detector.evaluate_single_reading(
        timestamp="2026-05-15 14:00:00",
        consumption_kwh=round(mean_14, 2),
        occupancy=25,
        temperature=26.0
    )
    assert normal_res["measured_kwh"] == round(mean_14, 2)
    assert normal_res["is_anomaly"] is False
    assert normal_res["waste_status"] == "NORMAL OPERATION"

    # 2. Extreme off-hours reading (Waste: night time, zero occupancy, massive draw)
    waste_res = detector.evaluate_single_reading(
        timestamp="2026-05-15 23:00:00",
        consumption_kwh=7.5, # Expected ~0.8
        occupancy=0,
        temperature=24.0,
        operating_start=8,
        operating_end=18
    )
    assert waste_res["is_anomaly"] is True
    assert "WASTE" in waste_res["waste_status"]
    assert waste_res["diff_kwh"] > 4.0


def test_waste_intelligence_engine_distinction():
    engine = EnergyWasteIntelligenceEngine(operating_start=8, operating_end=18, tariff_per_kwh=0.15)

    # Test Case A: Confirmed waste (8:30 PM, 0 occupancy, high consumption)
    case_a = engine.analyze_event(
        location_id="computer_lab",
        location_name="Computer Laboratory",
        timestamp="2026-05-15 20:30:00",
        consumption_kwh=4.8,
        expected_kwh=1.2,
        occupancy=0,
        temperature=25.0,
        is_anomaly=True,
        baseline_closed_kwh=0.8,
        baseline_open_kwh=4.5
    )
    assert case_a["is_waste"] is True
    assert "CONFIRMED ELECTRICITY WASTE" in case_a["status"]
    assert case_a["category"] == "OFF_HOURS_IDLE_WASTE"
    assert case_a["diff_kwh"] == 3.6
    assert case_a["estimated_extra_cost"] == round(3.6 * 0.15, 3)
    assert len(case_a["possible_explanations"]) > 0
    assert len(case_a["recommended_actions"]) > 0

    # Test Case B: High operational demand (2:00 PM, 35 occupants, high consumption)
    case_b = engine.analyze_event(
        location_id="computer_lab",
        location_name="Computer Laboratory",
        timestamp="2026-05-15 14:00:00",
        consumption_kwh=7.2,
        expected_kwh=4.5,
        occupancy=35,
        temperature=28.0,
        is_anomaly=True,
        baseline_closed_kwh=0.8,
        baseline_open_kwh=4.5
    )
    assert case_b["is_waste"] is False
    assert "ELEVATED OPERATIONAL DEMAND" in case_b["status"]

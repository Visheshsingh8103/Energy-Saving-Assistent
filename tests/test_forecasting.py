"""
tests/test_forecasting.py
Unit tests for machine learning electricity forecasting and baseline models.
"""

import os
import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.data_generator import generate_synthetic_energy_dataset
from src.forecasting import ElectricityForecaster, HourlyBaselineModel, build_forecasting_features


@pytest.fixture(scope="module")
def sample_dataset():
    # Generate 15 days of data for quick deterministic test
    df = generate_synthetic_energy_dataset(days=180, location_ids=["classroom_a"], seed=42)
    return df


def test_build_forecasting_features(sample_dataset):
    df_feat = build_forecasting_features(sample_dataset)
    expected_cols = ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos", "day_sin", "day_cos"]
    for col in expected_cols:
        assert col in df_feat.columns
    assert df_feat["hour_sin"].min() >= -1.0
    assert df_feat["hour_sin"].max() <= 1.0


def test_hourly_baseline_model(sample_dataset):
    df_feat = build_forecasting_features(sample_dataset)
    train_df = df_feat.iloc[:1000]
    test_df = df_feat.iloc[1000:1200]

    baseline = HourlyBaselineModel()
    baseline.fit(train_df[["hour", "is_weekend"]], train_df["consumption_kwh"])
    preds = baseline.predict(test_df[["hour", "is_weekend"]])

    assert len(preds) == len(test_df)
    assert (preds > 0).all()


def test_forecaster_train_and_save_load(sample_dataset, tmp_path):
    forecaster = ElectricityForecaster(location_id="test_loc")
    # Redirect models dir
    test_model_path = os.path.join(tmp_path, "forecaster_test_loc.joblib")
    forecaster.get_model_path = lambda: test_model_path

    metrics = forecaster.train_and_evaluate(sample_dataset, test_fraction=0.2, n_estimators=20)
    assert "random_forest" in metrics
    assert "baseline_model" in metrics
    assert metrics["random_forest"]["mae"] > 0
    assert metrics["random_forest"]["rmse"] > 0
    assert os.path.exists(test_model_path)

    # Test load
    new_forecaster = ElectricityForecaster(location_id="test_loc")
    new_forecaster.get_model_path = lambda: test_model_path
    loaded = new_forecaster.load()
    assert loaded is True
    assert new_forecaster.model is not None

    # Test 24-hour forward prediction
    forecast_df = new_forecaster.predict_next_24h()
    assert len(forecast_df) == 24
    assert "predicted_kwh" in forecast_df.columns
    assert "lower_bound_kwh" in forecast_df.columns
    assert "upper_bound_kwh" in forecast_df.columns
    assert (forecast_df["predicted_kwh"] >= 0).all()
    assert (forecast_df["upper_bound_kwh"] >= forecast_df["predicted_kwh"]).all()

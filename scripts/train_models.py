"""
scripts/train_models.py
CLI script to train Scikit-learn RandomForest electricity forecasting models
and IsolationForest anomaly detectors across all monitored facility locations.
"""

import sys
import os
import argparse
import pandas as pd

# Add repository root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import get_readings, get_locations, DEFAULT_DB_PATH
from src.forecasting import ElectricityForecaster
from src.anomaly_detection import EnergyAnomalyDetector


def main():
    parser = argparse.ArgumentParser(description="Train WattWise AI machine learning models.")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Path to SQLite database")
    parser.add_argument("--sensitivity", type=float, default=0.05, help="Anomaly detection contamination rate (0.01-0.15)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, args.db) if not os.path.isabs(args.db) else args.db

    print("==================================================")
    print(" WattWise AI — Model Training Pipeline")
    print("==================================================")

    locations = get_locations(db_path)
    if not locations:
        print("ERROR: No locations found in database. Run scripts/generate_data.py first.")
        sys.exit(1)

    print(f"Found {len(locations)} locations to train.")

    for loc in locations:
        loc_id = loc["id"]
        loc_name = loc["name"]
        print(f"\n--- Training for {loc_name} ({loc_id}) ---")

        df = get_readings(db_path=db_path, location_id=loc_id)
        if len(df) < 100:
            print(f"  [SKIPPED] Insufficient data ({len(df)} records). Need at least 100.")
            continue

        # 1. Forecasting Model
        print(f"  [1/2] Training RandomForestRegressor (Chronological train/test split)...")
        forecaster = ElectricityForecaster(location_id=loc_id)
        metrics = forecaster.train_and_evaluate(df, test_fraction=0.2)
        rf_mae = metrics["random_forest"]["mae"]
        rf_rmse = metrics["random_forest"]["rmse"]
        base_mae = metrics["baseline_model"]["mae"]
        base_rmse = metrics["baseline_model"]["rmse"]
        impr = metrics["improvement_pct"]["mae"]

        print(f"    -> RF Model Test MAE: {rf_mae:.4f} kWh | RMSE: {rf_rmse:.4f} kWh")
        print(f"    -> Hourly Baseline MAE: {base_mae:.4f} kWh | RMSE: {base_rmse:.4f} kWh")
        print(f"    -> RF Outperforms Baseline by {impr}%")
        print(f"    -> Saved model to: {forecaster.get_model_path()}")

        # 2. Anomaly Detection Model
        print(f"  [2/2] Training IsolationForest (Contamination = {args.sensitivity})...")
        detector = EnergyAnomalyDetector(location_id=loc_id, contamination=args.sensitivity)
        detector.fit(df)
        print(f"    -> IsolationForest trained on {detector.total_trained_samples:,} records.")
        print(f"    -> Saved detector to: {detector.get_model_path()}")

    print("\n==================================================")
    print(" Model training pipeline completed successfully!")
    print("==================================================")


if __name__ == "__main__":
    main()

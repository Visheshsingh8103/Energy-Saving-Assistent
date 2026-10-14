"""
scripts/generate_data.py
CLI script to generate reproducible synthetic energy dataset
and initialize the local SQLite database.
"""

import sys
import os
import argparse

# Add repository root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import save_sample_dataset_csv, generate_synthetic_energy_dataset
from src.database import init_db, insert_readings_bulk, get_database_summary, DEFAULT_DB_PATH


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic energy data for WattWise AI.")
    parser.add_argument("--days", type=int, default=180, help="Number of days of hourly telemetry (default: 180)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--csv", type=str, default="data/sample_energy_data.csv", help="Target CSV output path")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Target SQLite database path")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, args.csv)
    db_path = os.path.join(project_root, args.db) if not os.path.isabs(args.db) else args.db

    print(f"[1/3] Generating {args.days} days of synthetic energy data (Seed={args.seed})...")
    df = generate_synthetic_energy_dataset(days=args.days, seed=args.seed)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f" -> Saved {len(df):,} records to {csv_path}")

    print(f"[2/3] Initializing SQLite database schema at {db_path}...")
    init_db(db_path)

    print(f"[3/3] Loading synthetic readings into SQLite database...")
    inserted = insert_readings_bulk(db_path, df)
    print(f" -> Newly inserted records: {inserted:,}")

    summary = get_database_summary(db_path)
    print("\nDatabase Summary:")
    print(f"  Total records:      {summary['readings_count']:,}")
    print(f"  Locations:          {summary['locations_count']}")
    print(f"  Time span:          {summary['min_timestamp']} to {summary['max_timestamp']}")
    print(f"  Flagged anomalies:  {summary['anomalies_count']:,}")
    print("\nData generation complete!")


if __name__ == "__main__":
    main()

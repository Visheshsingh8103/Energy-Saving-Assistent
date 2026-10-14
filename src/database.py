"""
src/database.py
Database management module for WattWise AI using SQLite.
Handles connection, table creation, idempotent data loading,
settings persistence, and simulation event auditing.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wattwise.db")

SAMPLE_LOCATIONS = [
    {
        "id": "computer_lab",
        "name": "Computer Laboratory",
        "type": "computer_lab",
        "description": "High-density computing lab with 40 workstations, server rack, and dual AC units.",
        "baseline_kw_closed": 0.8,
        "baseline_kw_open": 4.5,
        "peak_kw": 8.5
    },
    {
        "id": "classroom_a",
        "name": "Classroom A",
        "type": "classroom",
        "description": "Large multimedia lecture hall with smartboard, projector, and 12 LED tube fixtures.",
        "baseline_kw_closed": 0.15,
        "baseline_kw_open": 1.4,
        "peak_kw": 2.8
    },
    {
        "id": "classroom_b",
        "name": "Classroom B",
        "type": "classroom",
        "description": "Standard seminar classroom with ceiling fans, projector, and peripheral outlets.",
        "baseline_kw_closed": 0.1,
        "baseline_kw_open": 1.1,
        "peak_kw": 2.2
    },
    {
        "id": "library",
        "name": "Library Reading Room",
        "type": "library",
        "description": "Silent study room with 8 catalog terminals, ambient lighting, and central ventilation.",
        "baseline_kw_closed": 0.5,
        "baseline_kw_open": 2.2,
        "peak_kw": 3.8
    },
    {
        "id": "faculty_office",
        "name": "Faculty Office Suite",
        "type": "faculty_office",
        "description": "Multi-desk faculty office with individual PCs, mini refrigerator, and desk lighting.",
        "baseline_kw_closed": 0.3,
        "baseline_kw_open": 1.8,
        "peak_kw": 3.0
    }
]

DEFAULT_SETTINGS = {
    "tariff_per_kwh": 0.15,
    "fixed_monthly_charge": 25.0,
    "tax_surcharge_rate": 0.05,
    "anomaly_sensitivity": 0.05,
    "operating_hours_start": 8,
    "operating_hours_end": 18,
    "co2_factor_kg_per_kwh": 0.82,
    "currency_symbol": "$"
}


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create and return a thread-safe connection to the SQLite database."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize SQLite database tables and default seed records."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS locations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT,
            baseline_kw_closed REAL NOT NULL,
            baseline_kw_open REAL NOT NULL,
            peak_kw REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS energy_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            location_id TEXT NOT NULL,
            hour INTEGER NOT NULL,
            day_of_week INTEGER NOT NULL,
            month INTEGER NOT NULL,
            temperature REAL NOT NULL,
            occupancy INTEGER NOT NULL,
            consumption_kwh REAL NOT NULL,
            is_anomaly_flag INTEGER DEFAULT 0,
            anomaly_score REAL DEFAULT 0.0,
            is_demo_injected INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (location_id) REFERENCES locations(id),
            UNIQUE(timestamp, location_id)
        );

        CREATE INDEX IF NOT EXISTS idx_readings_loc_ts 
            ON energy_readings(location_id, timestamp);

        CREATE TABLE IF NOT EXISTS user_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS simulation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            location_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            consumption_kwh REAL NOT NULL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (location_id) REFERENCES locations(id)
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id TEXT NOT NULL,
            title TEXT NOT NULL,
            reason TEXT NOT NULL,
            supporting_data TEXT,
            suggested_action TEXT NOT NULL,
            estimated_savings_kwh REAL DEFAULT 0.0,
            estimated_savings_cost REAL DEFAULT 0.0,
            confidence_level TEXT NOT NULL,
            status TEXT DEFAULT 'ACTIVE',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (location_id) REFERENCES locations(id)
        );

        CREATE TABLE IF NOT EXISTS waste_detective_findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            location_id TEXT NOT NULL,
            location_name TEXT NOT NULL,
            observed_kwh REAL NOT NULL,
            baseline_kwh REAL NOT NULL,
            diff_kwh REAL NOT NULL,
            cost_impact REAL NOT NULL,
            tariff_rate REAL NOT NULL,
            flag_reason TEXT NOT NULL,
            plausible_causes TEXT NOT NULL,
            recommended_action TEXT NOT NULL,
            evidence_score REAL NOT NULL,
            evidence_level TEXT NOT NULL,
            is_synthetic_test INTEGER DEFAULT 0,
            investigation_status TEXT DEFAULT 'PENDING',
            investigator_notes TEXT DEFAULT '',
            investigated_at TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (location_id) REFERENCES locations(id),
            UNIQUE(timestamp, location_id)
        );

        CREATE INDEX IF NOT EXISTS idx_detective_loc_ts 
            ON waste_detective_findings(location_id, timestamp);
    """)

    # Populate locations if empty
    cursor.execute("SELECT COUNT(*) FROM locations")
    if cursor.fetchone()[0] == 0:
        for loc in SAMPLE_LOCATIONS:
            cursor.execute(
                """INSERT INTO locations (id, name, type, description, baseline_kw_closed, baseline_kw_open, peak_kw)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (loc["id"], loc["name"], loc["type"], loc["description"],
                 loc["baseline_kw_closed"], loc["baseline_kw_open"], loc["peak_kw"])
            )

    # Populate default settings if empty
    for k, v in DEFAULT_SETTINGS.items():
        cursor.execute("SELECT COUNT(*) FROM user_settings WHERE key = ?", (k,))
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO user_settings (key, value) VALUES (?, ?)",
                (k, json.dumps(v))
            )

    conn.commit()
    conn.close()


def get_locations(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all monitored location records."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, type, description, baseline_kw_closed, baseline_kw_open, peak_kw FROM locations ORDER BY name")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_setting(db_path: str, key: str, default: Any = None) -> Any:
    """Retrieve a single setting parsed from JSON."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM user_settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return default
    try:
        return json.loads(row[0])
    except Exception:
        return row[0]


def set_setting(db_path: str, key: str, value: Any) -> None:
    """Upsert a setting value as JSON."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    serialized = json.dumps(value)
    cursor.execute(
        """INSERT INTO user_settings (key, value, updated_at) 
           VALUES (?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP""",
        (key, serialized)
    )
    conn.commit()
    conn.close()


def get_all_settings(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Retrieve all user settings as a dictionary."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM user_settings")
    rows = cursor.fetchall()
    conn.close()
    settings = dict(DEFAULT_SETTINGS)
    for row in rows:
        try:
            settings[row["key"]] = json.loads(row["value"])
        except Exception:
            settings[row["key"]] = row["value"]
    return settings


def insert_readings_bulk(db_path: str, df: pd.DataFrame) -> int:
    """
    Insert a batch of energy readings into the database.
    Ignores rows that duplicate (timestamp, location_id).
    Returns count of newly inserted rows.
    """
    if df.empty:
        return 0

    required_cols = [
        "timestamp", "location_id", "hour", "day_of_week", "month",
        "temperature", "occupancy", "consumption_kwh"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataframe: {col}")

    # Ensure optional columns
    is_anomaly = df["is_anomaly_flag"] if "is_anomaly_flag" in df.columns else [0] * len(df)
    score = df["anomaly_score"] if "anomaly_score" in df.columns else [0.0] * len(df)
    demo_injected = df["is_demo_injected"] if "is_demo_injected" in df.columns else [0] * len(df)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    records = []
    for idx, row in df.iterrows():
        records.append((
            str(row["timestamp"]),
            str(row["location_id"]),
            int(row["hour"]),
            int(row["day_of_week"]),
            int(row["month"]),
            float(row["temperature"]),
            int(row["occupancy"]),
            float(row["consumption_kwh"]),
            int(is_anomaly.iloc[idx] if hasattr(is_anomaly, "iloc") else is_anomaly[idx]),
            float(score.iloc[idx] if hasattr(score, "iloc") else score[idx]),
            int(demo_injected.iloc[idx] if hasattr(demo_injected, "iloc") else demo_injected[idx])
        ))

    cursor.executemany(
        """INSERT OR IGNORE INTO energy_readings 
           (timestamp, location_id, hour, day_of_week, month, temperature, occupancy,
            consumption_kwh, is_anomaly_flag, anomaly_score, is_demo_injected)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        records
    )
    conn.commit()
    inserted_count = cursor.rowcount
    conn.close()
    return inserted_count


def insert_single_reading(db_path: str, reading: Dict[str, Any]) -> int:
    """Insert a single real-time or demo energy reading."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT OR REPLACE INTO energy_readings 
           (timestamp, location_id, hour, day_of_week, month, temperature, occupancy,
            consumption_kwh, is_anomaly_flag, anomaly_score, is_demo_injected)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(reading["timestamp"]),
            str(reading["location_id"]),
            int(reading["hour"]),
            int(reading["day_of_week"]),
            int(reading["month"]),
            float(reading["temperature"]),
            int(reading["occupancy"]),
            float(reading["consumption_kwh"]),
            int(reading.get("is_anomaly_flag", 0)),
            float(reading.get("anomaly_score", 0.0)),
            int(reading.get("is_demo_injected", 0))
        )
    )
    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_readings(
    db_path: str = DEFAULT_DB_PATH,
    location_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """Query energy readings with optional filters."""
    conn = get_connection(db_path)
    query = """
        SELECT r.id, r.timestamp, r.location_id, l.name as location_name,
               r.hour, r.day_of_week, r.month, r.temperature, r.occupancy,
               r.consumption_kwh, r.is_anomaly_flag, r.anomaly_score, r.is_demo_injected
        FROM energy_readings r
        JOIN locations l ON r.location_id = l.id
        WHERE 1=1
    """
    params = []
    if location_id and location_id != "all":
        query += " AND r.location_id = ?"
        params.append(location_id)
    if start_date:
        query += " AND r.timestamp >= ?"
        params.append(str(start_date))
    if end_date:
        query += " AND r.timestamp <= ?"
        params.append(str(end_date))

    query += " ORDER BY r.timestamp ASC"
    if limit:
        query += f" LIMIT {int(limit)}"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def log_simulation_event(
    db_path: str,
    location_id: str,
    event_type: str,
    consumption_kwh: float,
    details: str
) -> int:
    """Log an injected anomaly or demo simulation event for auditing."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO simulation_events (timestamp, location_id, event_type, consumption_kwh, details)
           VALUES (?, ?, ?, ?, ?)""",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), location_id, event_type, consumption_kwh, details)
    )
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id


def get_simulation_events(db_path: str = DEFAULT_DB_PATH, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve audit log of simulation events."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT s.id, s.timestamp, s.location_id, l.name as location_name,
                  s.event_type, s.consumption_kwh, s.details, s.created_at
           FROM simulation_events s
           JOIN locations l ON s.location_id = l.id
           ORDER BY s.id DESC LIMIT ?""",
        (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def save_recommendations(db_path: str, recs: List[Dict[str, Any]]) -> None:
    """Save or refresh generated recommendations."""
    if not recs:
        return
    conn = get_connection(db_path)
    cursor = conn.cursor()
    for r in recs:
        cursor.execute(
            """INSERT INTO recommendations 
               (location_id, title, reason, supporting_data, suggested_action,
                estimated_savings_kwh, estimated_savings_cost, confidence_level)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["location_id"],
                r["title"],
                r["reason"],
                r.get("supporting_data", ""),
                r["suggested_action"],
                float(r.get("estimated_savings_kwh", 0.0)),
                float(r.get("estimated_savings_cost", 0.0)),
                r.get("confidence_level", "Medium")
            )
        )
    conn.commit()
    conn.close()


def get_recommendations(db_path: str = DEFAULT_DB_PATH, location_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve stored recommendations."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    if location_id and location_id != "all":
        cursor.execute(
            """SELECT r.*, l.name as location_name 
               FROM recommendations r
               JOIN locations l ON r.location_id = l.id
               WHERE r.location_id = ?
               ORDER BY r.id DESC LIMIT 100""",
            (location_id,)
        )
    else:
        cursor.execute(
            """SELECT r.*, l.name as location_name 
               FROM recommendations r
               JOIN locations l ON r.location_id = l.id
               ORDER BY r.id DESC LIMIT 100"""
        )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def save_detective_finding(db_path: str, finding: Dict[str, Any]) -> int:
    """
    Insert or update a waste detective finding.
    Preserves user investigation_status and investigator_notes if record already exists.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    causes = finding.get("plausible_causes", [])
    if isinstance(causes, (list, tuple)):
        causes_str = " | ".join(causes)
    else:
        causes_str = str(causes)

    cursor.execute(
        """INSERT INTO waste_detective_findings
           (timestamp, location_id, location_name, observed_kwh, baseline_kwh,
            diff_kwh, cost_impact, tariff_rate, flag_reason, plausible_causes,
            recommended_action, evidence_score, evidence_level, is_synthetic_test,
            investigation_status, investigator_notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(timestamp, location_id) DO UPDATE SET
            observed_kwh = excluded.observed_kwh,
            baseline_kwh = excluded.baseline_kwh,
            diff_kwh = excluded.diff_kwh,
            cost_impact = excluded.cost_impact,
            tariff_rate = excluded.tariff_rate,
            flag_reason = excluded.flag_reason,
            plausible_causes = excluded.plausible_causes,
            recommended_action = excluded.recommended_action,
            evidence_score = excluded.evidence_score,
            evidence_level = excluded.evidence_level,
            is_synthetic_test = excluded.is_synthetic_test""",
        (
            str(finding["timestamp"]),
            str(finding["location_id"]),
            str(finding.get("location_name", finding["location_id"])),
            float(finding["observed_kwh"]),
            float(finding["baseline_kwh"]),
            float(finding["diff_kwh"]),
            float(finding["cost_impact"]),
            float(finding.get("tariff_rate", 0.15)),
            str(finding.get("flag_reason", "")),
            causes_str,
            str(finding.get("recommended_action", "")),
            float(finding.get("evidence_score", 0.8)),
            str(finding.get("evidence_level", "High")),
            int(finding.get("is_synthetic_test", 0)),
            str(finding.get("investigation_status", "PENDING")),
            str(finding.get("investigator_notes", ""))
        )
    )
    last_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return last_id


def save_detective_findings_bulk(db_path: str, findings: List[Dict[str, Any]]) -> int:
    """Bulk insert detective findings."""
    count = 0
    for f in findings:
        save_detective_finding(db_path, f)
        count += 1
    return count


def get_detective_findings(
    db_path: str = DEFAULT_DB_PATH,
    location_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 150
) -> List[Dict[str, Any]]:
    """Retrieve detective findings with optional location and status filtering."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM waste_detective_findings WHERE 1=1"
    params = []

    if location_id and location_id != "all":
        query += " AND location_id = ?"
        params.append(location_id)

    if status_filter and status_filter != "all":
        query += " AND investigation_status = ?"
        params.append(status_filter)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(int(limit))

    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    # Expand plausible_causes back to list if split by pipe
    for r in rows:
        if isinstance(r.get("plausible_causes"), str):
            r["plausible_causes_list"] = [c.strip() for c in r["plausible_causes"].split(" | ") if c.strip()]
        else:
            r["plausible_causes_list"] = []

    return rows


def update_investigation_status(
    db_path: str,
    finding_id: int,
    new_status: str,
    investigator_notes: Optional[str] = None
) -> bool:
    """Update finding investigation status and auditor notes."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if investigator_notes is not None:
        cursor.execute(
            """UPDATE waste_detective_findings 
               SET investigation_status = ?, investigator_notes = ?, investigated_at = ?
               WHERE id = ?""",
            (new_status, investigator_notes, now_str, finding_id)
        )
    else:
        cursor.execute(
            """UPDATE waste_detective_findings 
               SET investigation_status = ?, investigated_at = ?
               WHERE id = ?""",
            (new_status, now_str, finding_id)
        )

    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def reset_demo_data(db_path: str = DEFAULT_DB_PATH) -> None:
    """Remove injected demo anomalies and simulation events, restoring pure synthetic baseline."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM energy_readings WHERE is_demo_injected = 1")
    cursor.execute("DELETE FROM simulation_events")
    cursor.execute("DELETE FROM recommendations")
    cursor.execute("DELETE FROM waste_detective_findings WHERE is_synthetic_test = 1")
    conn.commit()
    conn.close()


def get_database_summary(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Return database health metrics: total records, date range, locations."""
    if not os.path.exists(db_path):
        return {"exists": False, "readings_count": 0, "locations_count": 0}

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM energy_readings")
    readings_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM locations")
    locs_count = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM energy_readings")
    min_ts, max_ts = cursor.fetchone()

    cursor.execute("SELECT COUNT(*) FROM energy_readings WHERE is_anomaly_flag = 1")
    anomalies_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM simulation_events")
    events_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM waste_detective_findings")
    detective_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM waste_detective_findings WHERE investigation_status = 'PENDING'")
    pending_count = cursor.fetchone()[0]

    conn.close()

    return {
        "exists": True,
        "readings_count": readings_count,
        "locations_count": locs_count,
        "min_timestamp": min_ts,
        "max_timestamp": max_ts,
        "anomalies_count": anomalies_count,
        "simulation_events_count": events_count,
        "detective_findings_count": detective_count,
        "pending_investigations_count": pending_count
    }

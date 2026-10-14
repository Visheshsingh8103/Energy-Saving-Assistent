"""
tests/test_database.py
Unit tests for the SQLite persistence layer and audit system.
"""

import os
import tempfile
import pytest
import pandas as pd
from src.database import (
    init_db,
    get_locations,
    get_setting,
    set_setting,
    get_all_settings,
    insert_readings_bulk,
    insert_single_reading,
    get_readings,
    log_simulation_event,
    get_simulation_events,
    reset_demo_data,
    get_database_summary
)


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_init_and_locations(temp_db):
    locs = get_locations(temp_db)
    assert len(locs) == 5
    loc_ids = [l["id"] for l in locs]
    assert "computer_lab" in loc_ids
    assert "classroom_a" in loc_ids


def test_settings_persistence(temp_db):
    val = get_setting(temp_db, "tariff_per_kwh")
    assert val == 0.15

    set_setting(temp_db, "tariff_per_kwh", 0.22)
    new_val = get_setting(temp_db, "tariff_per_kwh")
    assert new_val == 0.22

    all_settings = get_all_settings(temp_db)
    assert all_settings["tariff_per_kwh"] == 0.22
    assert "operating_hours_start" in all_settings


def test_insert_readings_and_idempotence(temp_db):
    df = pd.DataFrame([
        {
            "timestamp": "2026-05-01 10:00:00",
            "location_id": "computer_lab",
            "hour": 10,
            "day_of_week": 4,
            "month": 5,
            "temperature": 25.5,
            "occupancy": 30,
            "consumption_kwh": 5.2,
            "is_anomaly_flag": 0,
            "anomaly_score": 0.0,
            "is_demo_injected": 0
        },
        {
            "timestamp": "2026-05-01 11:00:00",
            "location_id": "computer_lab",
            "hour": 11,
            "day_of_week": 4,
            "month": 5,
            "temperature": 26.2,
            "occupancy": 32,
            "consumption_kwh": 5.4,
            "is_anomaly_flag": 0,
            "anomaly_score": 0.0,
            "is_demo_injected": 0
        }
    ])

    inserted = insert_readings_bulk(temp_db, df)
    assert inserted == 2

    # Second insert of identical timestamps should be safely ignored (no duplication)
    inserted_again = insert_readings_bulk(temp_db, df)
    assert inserted_again == 0

    readings = get_readings(temp_db, location_id="computer_lab")
    assert len(readings) == 2


def test_simulation_events_and_reset(temp_db):
    # Insert demo injected reading
    single = {
        "timestamp": "2026-05-02 22:00:00",
        "location_id": "classroom_a",
        "hour": 22,
        "day_of_week": 5,
        "month": 5,
        "temperature": 23.0,
        "occupancy": 0,
        "consumption_kwh": 4.8,
        "is_anomaly_flag": 1,
        "anomaly_score": 0.88,
        "is_demo_injected": 1
    }
    insert_single_reading(temp_db, single)

    # Log event
    event_id = log_simulation_event(
        temp_db,
        location_id="classroom_a",
        event_type="TEST_INJECTION",
        consumption_kwh=4.8,
        details="Injected test anomaly for demonstration."
    )
    assert event_id > 0

    events = get_simulation_events(temp_db)
    assert len(events) == 1
    assert events[0]["event_type"] == "TEST_INJECTION"

    # Reset demo data
    reset_demo_data(temp_db)
    after_events = get_simulation_events(temp_db)
    assert len(after_events) == 0

    readings = get_readings(temp_db, location_id="classroom_a")
    assert len(readings) == 0

"""
tests/test_savings_calculator.py
Unit tests for the Energy Savings Lab and physics-based calculations.
"""

import pytest
from src.savings_calculator import (
    calculate_appliance_energy_kwh,
    simulate_room_savings,
    get_preset_scenario
)


def test_calculate_appliance_energy_kwh_standard():
    # 40 computers * 150 Watts * 10 hours = 60,000 Wh = 60.0 kWh
    kwh = calculate_appliance_energy_kwh(quantity=40, power_watts=150.0, operating_hours=10.0)
    assert kwh == 60.0


def test_calculate_appliance_energy_kwh_zero():
    assert calculate_appliance_energy_kwh(0, 100, 5) == 0.0
    assert calculate_appliance_energy_kwh(10, 0, 5) == 0.0
    assert calculate_appliance_energy_kwh(10, 100, 0) == 0.0


def test_calculate_appliance_energy_negative_raises():
    with pytest.raises(ValueError):
        calculate_appliance_energy_kwh(-1, 100, 5)
    with pytest.raises(ValueError):
        calculate_appliance_energy_kwh(10, -50, 5)
    with pytest.raises(ValueError):
        calculate_appliance_energy_kwh(10, 100, -2)


def test_simulate_room_savings_basic():
    baseline = {
        "num_computers": 10,
        "computer_watts": 200.0,
        "computer_hours": 10.0, # 20 kWh
        "num_lights": 10,
        "light_watts": 50.0,
        "light_hours": 10.0,    # 5 kWh
        "num_fans": 2,
        "fan_watts": 100.0,
        "fan_hours": 10.0,      # 2 kWh
        "other_kwh_daily": 0.0
    }
    # Baseline daily = 27 kWh

    proposed = {
        "num_computers": 10,
        "computer_watts": 50.0,
        "computer_hours": 8.0,  # 4 kWh
        "num_lights": 10,
        "light_watts": 15.0,
        "light_hours": 8.0,     # 1.2 kWh
        "num_fans": 2,
        "fan_watts": 35.0,
        "fan_hours": 8.0,       # 0.56 kWh
        "other_kwh_daily": 0.0
    }
    # Proposed daily = 5.76 kWh

    result = simulate_room_savings(
        baseline_params=baseline,
        proposed_params=proposed,
        tariff_per_kwh=0.20,
        co2_factor_kg_per_kwh=0.82,
        retrofit_cost=1000.0,
        days_per_month=30
    )

    assert result["is_estimate"] is True
    assert result["baseline"]["daily_kwh"] == 27.0
    assert result["baseline"]["monthly_kwh"] == 810.0
    assert result["baseline"]["monthly_cost"] == 162.0

    assert result["proposed"]["daily_kwh"] == 5.76
    assert result["proposed"]["monthly_kwh"] == 172.8
    assert result["proposed"]["monthly_cost"] == 34.56

    assert result["savings"]["daily_kwh"] == 21.24
    assert result["savings"]["monthly_kwh"] == 637.2
    assert result["savings"]["monthly_cost"] == 127.44
    assert result["savings"]["percent_reduction"] == pytest.approx(78.7, 0.1)

    # Simple payback in months = 1000 / 127.44 = ~7.8 months
    assert result["savings"]["payback_months"] == pytest.approx(7.8, 0.1)
    assert result["savings"]["monthly_co2_kg"] == pytest.approx(637.2 * 0.82, 0.1)


def test_presets():
    preset = get_preset_scenario("computer_lab_baseline")
    assert preset["num_computers"] == 40
    assert preset["computer_watts"] > 0

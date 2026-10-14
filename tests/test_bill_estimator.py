"""
tests/test_bill_estimator.py
Unit tests for the Electricity Bill Estimator module.
"""

import pytest
import pandas as pd
from src.bill_estimator import calculate_electricity_bill, estimate_bill_from_dataframe


def test_calculate_electricity_bill_standard():
    # 1000 kWh @ $0.15 = $150
    # Fixed charge: $25 (30 days)
    # Demand charge: 0
    # Subtotal: $175
    # Tax @ 5%: $8.75
    # Total: $183.75
    bill = calculate_electricity_bill(
        total_kwh=1000.0,
        tariff_per_kwh=0.15,
        fixed_monthly_charge=25.0,
        tax_surcharge_rate=0.05,
        billing_period_days=30
    )

    assert bill["total_bill"] == 183.75
    assert bill["itemized_charges"]["energy_charge"] == 150.0
    assert bill["itemized_charges"]["fixed_charge"] == 25.0
    assert bill["itemized_charges"]["subtotal"] == 175.0
    assert bill["itemized_charges"]["tax_and_surcharge"] == 8.75
    assert bill["effective_rate_per_kwh"] == pytest.approx(0.1838, 0.001)


def test_calculate_bill_with_demand_charge():
    bill = calculate_electricity_bill(
        total_kwh=500.0,
        tariff_per_kwh=0.10,
        fixed_monthly_charge=20.0,
        tax_surcharge_rate=0.10,
        peak_kw=15.0,
        demand_charge_per_kw=4.0, # 15 * 4 = 60
        billing_period_days=30
    )
    # Energy: 500 * 0.10 = $50
    # Fixed: $20
    # Demand: $60
    # Subtotal: $130
    # Tax @ 10%: $13
    # Total: $143
    assert bill["itemized_charges"]["energy_charge"] == 50.0
    assert bill["itemized_charges"]["demand_charge"] == 60.0
    assert bill["total_bill"] == 143.0


def test_bill_comparison_period():
    bill = calculate_electricity_bill(
        total_kwh=1200.0,
        tariff_per_kwh=0.15,
        fixed_monthly_charge=25.0,
        tax_surcharge_rate=0.05,
        previous_period_kwh=1000.0
    )
    cmp = bill["comparison"]
    assert cmp is not None
    assert cmp["previous_kwh"] == 1000.0
    assert cmp["kwh_difference"] == 200.0
    assert cmp["percent_change"] == 20.0
    assert cmp["trend"] == "INCREASED"


def test_bill_negative_values_raise():
    with pytest.raises(ValueError):
        calculate_electricity_bill(total_kwh=-10.0)
    with pytest.raises(ValueError):
        calculate_electricity_bill(total_kwh=100.0, tariff_per_kwh=-0.1)


def test_estimate_bill_from_dataframe():
    df = pd.DataFrame({
        "timestamp": ["2026-01-01 10:00:00", "2026-01-01 11:00:00"],
        "consumption_kwh": [10.5, 9.5]
    })
    bill = estimate_bill_from_dataframe(df, tariff_per_kwh=0.20, fixed_monthly_charge=0.0, tax_surcharge_rate=0.0)
    assert bill["total_kwh"] == 20.0
    assert bill["total_bill"] == 4.0

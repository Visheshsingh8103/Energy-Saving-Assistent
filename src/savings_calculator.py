"""
src/savings_calculator.py
Energy Savings Lab module for WattWise AI.
Performs deterministic what-if simulations of appliance power consumption,
efficiency upgrades, operating hour reductions, and financial returns.

All outputs from this module are clearly designated as theoretical estimates
governed by physics formulas and user-specified assumptions.
"""

from typing import Dict, Any, List, Optional


def calculate_appliance_energy_kwh(quantity: int, power_watts: float, operating_hours: float) -> float:
    """
    Calculate energy consumption for an appliance group in kilowatt-hours.
    Formula: Energy (kWh) = (Quantity * Power_Watts * Operating_Hours) / 1000
    """
    if quantity < 0 or power_watts < 0 or operating_hours < 0:
        raise ValueError("Appliance quantity, power, and operating hours must be non-negative.")
    
    return round((quantity * power_watts * operating_hours) / 1000.0, 4)


def simulate_room_savings(
    baseline_params: Dict[str, Any],
    proposed_params: Dict[str, Any],
    tariff_per_kwh: float = 0.15,
    co2_factor_kg_per_kwh: float = 0.82,
    retrofit_cost: float = 0.0,
    days_per_month: int = 30
) -> Dict[str, Any]:
    """
    Perform a comparative what-if energy savings simulation between a baseline scenario
    and a proposed energy conservation scenario.

    Parameters:
        baseline_params: Dict with appliance counts, wattages, and hours:
            - num_computers, computer_watts, computer_hours
            - num_lights, light_watts, light_hours
            - num_fans, fan_watts, fan_hours
            - optional: other_appliance_kwh_daily
        proposed_params: Dict with modified parameters for the proposed scenario.
        tariff_per_kwh: Price of electricity per kWh.
        co2_factor_kg_per_kwh: Average grid carbon intensity in kg CO2 per kWh.
        retrofit_cost: Optional upfront investment cost for upgrades to compute payback.
        days_per_month: Standard billing days per month (default 30).

    Returns:
        Dictionary containing granular and aggregated baseline, proposed, and savings metrics.
    """
    if tariff_per_kwh < 0:
        raise ValueError("Tariff per kWh must be non-negative.")

    # 1. Baseline Calculations
    base_comp_kwh = calculate_appliance_energy_kwh(
        baseline_params.get("num_computers", 0),
        baseline_params.get("computer_watts", 0.0),
        baseline_params.get("computer_hours", 0.0)
    )
    base_light_kwh = calculate_appliance_energy_kwh(
        baseline_params.get("num_lights", 0),
        baseline_params.get("light_watts", 0.0),
        baseline_params.get("light_hours", 0.0)
    )
    base_fan_kwh = calculate_appliance_energy_kwh(
        baseline_params.get("num_fans", 0),
        baseline_params.get("fan_watts", 0.0),
        baseline_params.get("fan_hours", 0.0)
    )
    base_other_kwh = float(baseline_params.get("other_kwh_daily", 0.0))

    baseline_daily_kwh = round(base_comp_kwh + base_light_kwh + base_fan_kwh + base_other_kwh, 3)
    baseline_monthly_kwh = round(baseline_daily_kwh * days_per_month, 2)
    baseline_monthly_cost = round(baseline_monthly_kwh * tariff_per_kwh, 2)
    baseline_annual_cost = round(baseline_monthly_cost * 12.0, 2)

    # 2. Proposed Calculations
    prop_comp_kwh = calculate_appliance_energy_kwh(
        proposed_params.get("num_computers", 0),
        proposed_params.get("computer_watts", 0.0),
        proposed_params.get("computer_hours", 0.0)
    )
    prop_light_kwh = calculate_appliance_energy_kwh(
        proposed_params.get("num_lights", 0),
        proposed_params.get("light_watts", 0.0),
        proposed_params.get("light_hours", 0.0)
    )
    prop_fan_kwh = calculate_appliance_energy_kwh(
        proposed_params.get("num_fans", 0),
        proposed_params.get("fan_watts", 0.0),
        proposed_params.get("fan_hours", 0.0)
    )
    prop_other_kwh = float(proposed_params.get("other_kwh_daily", 0.0))

    proposed_daily_kwh = round(prop_comp_kwh + prop_light_kwh + prop_fan_kwh + prop_other_kwh, 3)
    proposed_monthly_kwh = round(proposed_daily_kwh * days_per_month, 2)
    proposed_monthly_cost = round(proposed_monthly_kwh * tariff_per_kwh, 2)
    proposed_annual_cost = round(proposed_monthly_cost * 12.0, 2)

    # 3. Savings Calculations
    daily_savings_kwh = max(0.0, round(baseline_daily_kwh - proposed_daily_kwh, 3))
    monthly_savings_kwh = max(0.0, round(baseline_monthly_kwh - proposed_monthly_kwh, 2))
    annual_savings_kwh = max(0.0, round(monthly_savings_kwh * 12.0, 2))

    monthly_savings_cost = max(0.0, round(baseline_monthly_cost - proposed_monthly_cost, 2))
    annual_savings_cost = max(0.0, round(baseline_annual_cost - proposed_annual_cost, 2))

    # Percentage reduction
    if baseline_daily_kwh > 0:
        percent_reduction = round((daily_savings_kwh / baseline_daily_kwh) * 100.0, 1)
    else:
        percent_reduction = 0.0

    # Carbon reduction (kg CO2)
    monthly_co2_kg = round(monthly_savings_kwh * co2_factor_kg_per_kwh, 1)
    annual_co2_kg = round(annual_savings_kwh * co2_factor_kg_per_kwh, 1)
    annual_trees_equivalent = round(annual_co2_kg / 21.77, 1) # ~21.77 kg CO2 absorbed per tree per year

    # Financial Payback Period (Simple Payback in Months)
    if retrofit_cost > 0 and monthly_savings_cost > 0:
        payback_months = round(retrofit_cost / monthly_savings_cost, 1)
    elif retrofit_cost > 0:
        payback_months = float("inf")
    else:
        payback_months = 0.0

    return {
        "is_estimate": True,
        "disclaimer": "All values are theoretical projections based on user-provided appliance ratings and hours.",
        "baseline": {
            "daily_kwh": baseline_daily_kwh,
            "monthly_kwh": baseline_monthly_kwh,
            "monthly_cost": baseline_monthly_cost,
            "annual_cost": baseline_annual_cost,
            "appliance_breakdown_daily_kwh": {
                "computers": base_comp_kwh,
                "lights": base_light_kwh,
                "fans": base_fan_kwh,
                "other": base_other_kwh
            }
        },
        "proposed": {
            "daily_kwh": proposed_daily_kwh,
            "monthly_kwh": proposed_monthly_kwh,
            "monthly_cost": proposed_monthly_cost,
            "annual_cost": proposed_annual_cost,
            "appliance_breakdown_daily_kwh": {
                "computers": prop_comp_kwh,
                "lights": prop_light_kwh,
                "fans": prop_fan_kwh,
                "other": prop_other_kwh
            }
        },
        "savings": {
            "daily_kwh": daily_savings_kwh,
            "monthly_kwh": monthly_savings_kwh,
            "annual_kwh": annual_savings_kwh,
            "monthly_cost": monthly_savings_cost,
            "annual_cost": annual_savings_cost,
            "percent_reduction": percent_reduction,
            "monthly_co2_kg": monthly_co2_kg,
            "annual_co2_kg": annual_co2_kg,
            "annual_trees_equivalent": annual_trees_equivalent,
            "retrofit_cost": retrofit_cost,
            "payback_months": payback_months
        }
    }


def get_preset_scenario(scenario_key: str) -> Dict[str, Any]:
    """Retrieve standard benchmark scenarios for educational demonstrations."""
    presets = {
        "computer_lab_baseline": {
            "name": "Standard Computer Lab (Baseline)",
            "num_computers": 40,
            "computer_watts": 150.0, # Desktop towers
            "computer_hours": 10.0,
            "num_lights": 16,
            "light_watts": 40.0,     # Fluorescent T8 tubes
            "light_hours": 12.0,
            "num_fans": 6,
            "fan_watts": 75.0,
            "fan_hours": 12.0,
            "other_kwh_daily": 3.0   # Switchboard, network switch
        },
        "computer_lab_retrofitted": {
            "name": "Eco-Optimized Computer Lab (Proposed)",
            "num_computers": 40,
            "computer_watts": 45.0,  # Energy Star Mini PCs
            "computer_hours": 7.5,   # Auto-sleep after classes
            "num_lights": 16,
            "light_watts": 14.0,     # High-efficiency LED tubes
            "light_hours": 8.0,      # Daylight harvesting / occupancy sensor
            "num_fans": 6,
            "fan_watts": 35.0,       # BLDC energy-efficient fans
            "fan_hours": 8.0,
            "other_kwh_daily": 1.5   # Smart power strips
        },
        "classroom_baseline": {
            "name": "Typical Classroom (Baseline)",
            "num_computers": 1,
            "computer_watts": 120.0,
            "computer_hours": 8.0,
            "num_lights": 12,
            "light_watts": 40.0,
            "light_hours": 9.0,
            "num_fans": 4,
            "fan_watts": 75.0,
            "fan_hours": 9.0,
            "other_kwh_daily": 1.0
        },
        "classroom_retrofitted": {
            "name": "Green Classroom (Proposed)",
            "num_computers": 1,
            "computer_watts": 30.0,  # Laptop
            "computer_hours": 5.0,
            "num_lights": 12,
            "light_watts": 12.0,     # LED batten
            "light_hours": 6.0,
            "num_fans": 4,
            "fan_watts": 28.0,       # 5-star BLDC
            "fan_hours": 6.0,
            "other_kwh_daily": 0.5
        }
    }
    return presets.get(scenario_key, presets["computer_lab_baseline"])

"""
src/bill_estimator.py
Electricity Bill Estimator module for WattWise AI.
Provides transparent calculation of utility costs with configurable tariffs,
fixed grid connection charges, local surcharges/taxes, and historical period comparisons.
"""

from typing import Dict, Any, Optional
import pandas as pd


def calculate_electricity_bill(
    total_kwh: float,
    tariff_per_kwh: float = 0.15,
    fixed_monthly_charge: float = 25.0,
    tax_surcharge_rate: float = 0.05,
    peak_kw: Optional[float] = None,
    demand_charge_per_kw: float = 0.0,
    billing_period_days: int = 30,
    previous_period_kwh: Optional[float] = None,
    currency_symbol: str = "$"
) -> Dict[str, Any]:
    """
    Calculate an itemized electricity bill estimate.

    Parameters:
        total_kwh: Total energy consumed in kilowatt-hours.
        tariff_per_kwh: Base volumetric energy rate per kWh.
        fixed_monthly_charge: Base monthly customer/meter connection fee.
        tax_surcharge_rate: Decimal fraction for municipal/state taxes and environmental fees (e.g. 0.05 = 5%).
        peak_kw: Peak demand measured in kW during the billing interval.
        demand_charge_per_kw: Capacity/demand tariff applied per kW of peak demand.
        billing_period_days: Number of days in the billing window (pro-rates fixed charges).
        previous_period_kwh: Energy consumption from previous comparable period for variance analysis.
        currency_symbol: Currency display symbol.

    Returns:
        Structured breakdown of energy charges, demand charges, fixed fees, taxes, and period-over-period delta.
    """
    if total_kwh < 0 or tariff_per_kwh < 0 or fixed_monthly_charge < 0:
        raise ValueError("Energy consumption and charges cannot be negative.")

    # Pro-rate fixed charge based on 30-day standard month
    prorated_fixed_charge = round(fixed_monthly_charge * (billing_period_days / 30.0), 2)

    # Volumetric energy charge
    energy_charge = round(total_kwh * tariff_per_kwh, 2)

    # Demand charge (if applicable)
    demand_charge = 0.0
    if peak_kw is not None and peak_kw > 0 and demand_charge_per_kw > 0:
        demand_charge = round(peak_kw * demand_charge_per_kw, 2)

    # Subtotal
    subtotal = round(energy_charge + prorated_fixed_charge + demand_charge, 2)

    # Taxes and regulatory surcharges
    tax_amount = round(subtotal * max(0.0, tax_surcharge_rate), 2)

    # Grand total
    total_bill = round(subtotal + tax_amount, 2)

    # Effective cost per kWh (all-in)
    effective_rate_per_kwh = round(total_bill / total_kwh, 4) if total_kwh > 0 else tariff_per_kwh

    # Comparison with previous period
    comparison = None
    if previous_period_kwh is not None and previous_period_kwh >= 0:
        prev_energy_charge = round(previous_period_kwh * tariff_per_kwh, 2)
        prev_subtotal = round(prev_energy_charge + prorated_fixed_charge, 2)
        prev_total = round(prev_subtotal * (1.0 + max(0.0, tax_surcharge_rate)), 2)

        kwh_diff = round(total_kwh - previous_period_kwh, 2)
        cost_diff = round(total_bill - prev_total, 2)
        pct_change = round((kwh_diff / previous_period_kwh) * 100.0, 1) if previous_period_kwh > 0 else 0.0

        comparison = {
            "previous_kwh": previous_period_kwh,
            "previous_total_bill": prev_total,
            "kwh_difference": kwh_diff,
            "cost_difference": cost_diff,
            "percent_change": pct_change,
            "trend": "INCREASED" if kwh_diff > 0 else ("DECREASED" if kwh_diff < 0 else "FLAT")
        }

    return {
        "is_estimate": True,
        "disclaimer": (
            "Estimates do not account for dynamic Time-of-Use (TOU) tier variations, "
            "seasonal utility adjustments, or power factor penalties."
        ),
        "currency": currency_symbol,
        "billing_period_days": billing_period_days,
        "total_kwh": round(total_kwh, 2),
        "tariff_per_kwh": tariff_per_kwh,
        "itemized_charges": {
            "energy_charge": energy_charge,
            "fixed_charge": prorated_fixed_charge,
            "demand_charge": demand_charge,
            "subtotal": subtotal,
            "tax_and_surcharge": tax_amount,
            "tax_rate_percent": round(tax_surcharge_rate * 100.0, 2)
        },
        "total_bill": total_bill,
        "effective_rate_per_kwh": effective_rate_per_kwh,
        "comparison": comparison
    }


def estimate_bill_from_dataframe(
    df: pd.DataFrame,
    tariff_per_kwh: float = 0.15,
    fixed_monthly_charge: float = 25.0,
    tax_surcharge_rate: float = 0.05,
    demand_charge_per_kw: float = 0.0,
    currency_symbol: str = "$"
) -> Dict[str, Any]:
    """Calculate bill estimate directly from a DataFrame of readings."""
    if df.empty or "consumption_kwh" not in df.columns:
        return calculate_electricity_bill(
            total_kwh=0.0,
            tariff_per_kwh=tariff_per_kwh,
            fixed_monthly_charge=fixed_monthly_charge,
            tax_surcharge_rate=tax_surcharge_rate,
            currency_symbol=currency_symbol
        )

    total_kwh = float(df["consumption_kwh"].sum())
    
    # Peak demand approx (max hourly consumption in kW)
    peak_kw = float(df["consumption_kwh"].max())

    # Estimate billing period in days from timestamps
    days = 30
    if "timestamp" in df.columns and len(df) > 1:
        try:
            ts_series = pd.to_datetime(df["timestamp"])
            delta_days = (ts_series.max() - ts_series.min()).total_seconds() / 86400.0
            if delta_days >= 1.0:
                days = int(round(delta_days))
        except Exception:
            days = 30

    return calculate_electricity_bill(
        total_kwh=total_kwh,
        tariff_per_kwh=tariff_per_kwh,
        fixed_monthly_charge=fixed_monthly_charge,
        tax_surcharge_rate=tax_surcharge_rate,
        peak_kw=peak_kw,
        demand_charge_per_kw=demand_charge_per_kw,
        billing_period_days=days,
        currency_symbol=currency_symbol
    )

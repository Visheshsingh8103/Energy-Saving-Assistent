"""
src/recommendations.py
Intelligent Recommendation Engine for WattWise AI.
Analyzes historical and current consumption patterns using deterministic rules
to deliver prioritized energy-saving recommendations with quantitative savings estimates.
"""

from typing import List, Dict, Any, Optional
import pandas as pd


class RecommendationEngine:
    """
    Deterministic rule-based advisory engine that transforms empirical energy telemetry
    into prioritized, cost-quantified facility optimization actions.
    """
    def __init__(
        self,
        operating_start: int = 8,
        operating_end: int = 18,
        tariff_per_kwh: float = 0.15
    ):
        self.operating_start = operating_start
        self.operating_end = operating_end
        self.tariff_per_kwh = tariff_per_kwh

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        locations_meta: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate full dataset telemetry across rooms and generate prioritized recommendations.
        """
        if df.empty:
            return []

        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy["timestamp"]):
            df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

        df_copy["hour"] = df_copy["timestamp"].dt.hour
        df_copy["day_of_week"] = df_copy["timestamp"].dt.dayofweek
        df_copy["is_weekend"] = (df_copy["day_of_week"] >= 5).astype(int)

        recommendations = []
        locations = df_copy["location_id"].unique()

        for loc_id in locations:
            loc_df = df_copy[df_copy["location_id"] == loc_id]
            loc_name = loc_df["location_name"].iloc[0] if "location_name" in loc_df.columns else loc_id
            meta = (locations_meta or {}).get(loc_id, {})
            base_closed = meta.get("baseline_kw_closed", 0.5)

            # Rule 1: Off-Hours Inactive Consumption
            off_hours_df = loc_df[
                ((loc_df["hour"] < self.operating_start) | (loc_df["hour"] >= self.operating_end)) &
                (loc_df["occupancy"] == 0)
            ]
            if not off_hours_df.empty:
                avg_off_draw = off_hours_df["consumption_kwh"].mean()
                if avg_off_draw > (base_closed * 1.5):
                    excess_hourly = max(0.0, avg_off_draw - base_closed)
                    # 14 off-hours per day * 30 days
                    monthly_excess_kwh = round(excess_hourly * 14 * 30, 1)
                    monthly_cost = round(monthly_excess_kwh * self.tariff_per_kwh, 2)

                    recommendations.append({
                        "location_id": loc_id,
                        "location_name": loc_name,
                        "title": f"Automate Night Shutdown for {loc_name}",
                        "reason": f"Off-hours draw averages {avg_off_draw:.2f} kWh/hr when unoccupied, exceeding base expectation of {base_closed:.2f} kWh/hr.",
                        "supporting_data": f"Off-hours idle draw: {avg_off_draw:.2f} kWh/hr vs baseline {base_closed:.2f} kWh/hr across {len(off_hours_df)} audited hours.",
                        "suggested_action": "Install smart programmable relays or configure automated OS sleep schedules to power down peripherals and HVAC between 18:00 and 08:00.",
                        "estimated_savings_kwh": monthly_excess_kwh,
                        "estimated_savings_cost": monthly_cost,
                        "confidence_level": "High"
                    })

            # Rule 2: High Daytime Peak Reduction
            peak_reading = loc_df["consumption_kwh"].max()
            mean_reading = loc_df["consumption_kwh"].mean()
            if peak_reading > (mean_reading * 2.5) and peak_reading > 3.0:
                peak_hours = loc_df[loc_df["consumption_kwh"] > (mean_reading * 2.0)]
                potential_kwh = round((peak_reading - mean_reading * 1.5) * len(peak_hours) * 0.5, 1)
                potential_cost = round(potential_kwh * self.tariff_per_kwh, 2)

                recommendations.append({
                    "location_id": loc_id,
                    "location_name": loc_name,
                    "title": f"Stagger Heavy Equipment Loads in {loc_name}",
                    "reason": f"Observed peak power draw reached {peak_reading:.2f} kWh, which is {peak_reading / max(0.1, mean_reading):.1f}x higher than average.",
                    "supporting_data": f"Peak demand: {peak_reading:.2f} kWh | Average demand: {mean_reading:.2f} kWh | High-draw occurrences: {len(peak_hours)}.",
                    "suggested_action": "Avoid powering all computer workstations or high-load heating/cooling appliances simultaneously. Stagger boot cycles by 5-10 minutes.",
                    "estimated_savings_kwh": potential_kwh,
                    "estimated_savings_cost": potential_cost,
                    "confidence_level": "Medium"
                })

            # Rule 3: High Temperature HVAC Optimization
            hot_days_df = loc_df[loc_df["temperature"] >= 30.0]
            if not hot_days_df.empty:
                hot_mean = hot_days_df["consumption_kwh"].mean()
                mild_days_df = loc_df[(loc_df["temperature"] >= 20.0) & (loc_df["temperature"] <= 24.0)]
                mild_mean = mild_days_df["consumption_kwh"].mean() if not mild_days_df.empty else (hot_mean * 0.75)

                if hot_mean > (mild_mean * 1.25):
                    # Raising thermostat by 1°C saves ~6% on cooling
                    cooling_portion = hot_mean - mild_mean
                    est_savings_kwh = round(cooling_portion * 0.15 * len(hot_days_df), 1)
                    est_savings_cost = round(est_savings_kwh * self.tariff_per_kwh, 2)

                    recommendations.append({
                        "location_id": loc_id,
                        "location_name": loc_name,
                        "title": f"Calibrate HVAC Setpoint & Ingress in {loc_name}",
                        "reason": f"Energy usage increases by {((hot_mean - mild_mean) / max(0.1, mild_mean)) * 100:.1f}% during ambient temperatures above 30°C.",
                        "supporting_data": f"High temp average: {hot_mean:.2f} kWh vs moderate temp average: {mild_mean:.2f} kWh.",
                        "suggested_action": "Adjust target room temperature from 21°C to 24°C, inspect window weatherstripping, and draw thermal blinds during peak sun angles.",
                        "estimated_savings_kwh": est_savings_kwh,
                        "estimated_savings_cost": est_savings_cost,
                        "confidence_level": "Medium"
                    })

            # Rule 4: Frequent Anomaly Cluster Warning
            anom_count = loc_df["is_anomaly_flag"].sum() if "is_anomaly_flag" in loc_df.columns else 0
            if anom_count > 15:
                recommendations.append({
                    "location_id": loc_id,
                    "location_name": loc_name,
                    "title": f"Hardware & Wiring Audit for {loc_name}",
                    "reason": f"IsolationForest flagged {anom_count} anomalous readings over the monitored timeframe.",
                    "supporting_data": f"Total flagged anomalous events: {anom_count} (Rate: {(anom_count / len(loc_df)) * 100:.1f}%).",
                    "suggested_action": "Check sub-metering current transformers, assess capacitor bank health, and inspect power distribution boards for loose terminals.",
                    "estimated_savings_kwh": round(anom_count * 1.5, 1),
                    "estimated_savings_cost": round(anom_count * 1.5 * self.tariff_per_kwh, 2),
                    "confidence_level": "High"
                })

        # Sort recommendations by estimated cost savings descending
        recommendations.sort(key=lambda r: r.get("estimated_savings_cost", 0.0), reverse=True)
        return recommendations

"""
src/anomaly_detection.py
AI Anomaly Detection module for WattWise AI using Scikit-learn IsolationForest.
Detects statistical deviations in electricity consumption while distinguishing
statistical anomalies from confirmed energy waste.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import IsolationForest

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

ANOMALY_FEATURE_COLS = [
    "consumption_kwh", "hour", "day_of_week", "occupancy", "temperature",
    "diff_from_hourly_mean", "ratio_to_hourly_mean"
]


def extract_anomaly_features(df: pd.DataFrame, hourly_means: Optional[Dict[int, float]] = None) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """
    Extract statistical and context features for anomaly detection.
    Computes hour-of-day expected mean consumption.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Compute or apply baseline hourly means
    if hourly_means is None:
        hourly_means = df.groupby("hour")["consumption_kwh"].mean().to_dict()

    df["hourly_mean_baseline"] = df["hour"].map(hourly_means).fillna(df["consumption_kwh"].mean())
    df["diff_from_hourly_mean"] = df["consumption_kwh"] - df["hourly_mean_baseline"]
    # Safe ratio avoiding division by zero
    df["ratio_to_hourly_mean"] = df["consumption_kwh"] / np.maximum(0.05, df["hourly_mean_baseline"])

    if "occupancy" not in df.columns:
        df["occupancy"] = 0
    if "temperature" not in df.columns:
        df["temperature"] = 24.0

    return df, hourly_means


class EnergyAnomalyDetector:
    """
    Anomaly detection pipeline using Scikit-Learn IsolationForest.
    Provides configurable sensitivity and contextual investigation insights.
    """
    def __init__(self, location_id: str = "general", contamination: float = 0.05):
        self.location_id = location_id
        self.contamination = max(0.01, min(0.20, float(contamination)))
        self.model: Optional[IsolationForest] = None
        self.hourly_means: Dict[int, float] = {}
        self.trained_at: Optional[str] = None
        self.total_trained_samples: int = 0

    def get_model_path(self) -> str:
        os.makedirs(MODELS_DIR, exist_ok=True)
        return os.path.join(MODELS_DIR, f"anomaly_detector_{self.location_id}.joblib")

    def fit(self, df: pd.DataFrame) -> "EnergyAnomalyDetector":
        """Fit IsolationForest on historical readings."""
        if len(df) < 50:
            raise ValueError(f"Insufficient data for anomaly detector ({len(df)} rows). Need at least 50.")

        df_feat, self.hourly_means = extract_anomaly_features(df)
        X = df_feat[ANOMALY_FEATURE_COLS].fillna(0)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples="auto"
        )
        self.model.fit(X)

        self.total_trained_samples = len(X)
        self.trained_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        return self

    def predict(self, df: pd.DataFrame, tariff_per_kwh: float = 0.15) -> pd.DataFrame:
        """
        Flag anomalies in a dataset and calculate difference from expected consumption.
        IsolationForest returns -1 for anomaly, 1 for normal.
        """
        if self.model is None:
            if not self.load():
                self.fit(df)

        df_feat, _ = extract_anomaly_features(df, self.hourly_means)
        X = df_feat[ANOMALY_FEATURE_COLS].fillna(0)

        # Raw scores: lower is more anomalous
        raw_scores = self.model.decision_function(X)
        raw_preds = self.model.predict(X) # -1 = anomaly, 1 = normal

        result_df = df.copy()
        result_df["is_anomaly_flag"] = (raw_preds == -1).astype(int)
        # Normalize score between 0 (normal) and 1 (highly anomalous)
        # Decision function is negative for anomalies
        result_df["anomaly_score"] = np.round(np.clip((0.15 - raw_scores) / 0.35, 0.0, 1.0), 3)

        result_df["expected_kwh"] = df_feat["hourly_mean_baseline"].round(3)
        result_df["diff_kwh"] = (result_df["consumption_kwh"] - result_df["expected_kwh"]).round(3)
        result_df["cost_diff"] = (result_df["diff_kwh"] * tariff_per_kwh).round(3)

        # Distinguish statistical anomaly from verified waste vs high operational load
        investigations = []
        for _, row in result_df.iterrows():
            if row["is_anomaly_flag"] == 1:
                diff = row["diff_kwh"]
                occ = row.get("occupancy", 0)
                hr = row["hour"] if "hour" in row else 12

                if diff > 0 and occ == 0 and (hr < 8 or hr >= 18):
                    inv = "High Off-Hours Draw: Zero occupancy detected. Likely equipment left running."
                elif diff > 0 and occ > 15:
                    inv = "High Operational Demand: Occupancy is high. May be legitimate intensive lab activity."
                elif diff > 0 and occ == 0:
                    inv = "Unoccupied Consumption: Room is vacant during operating hours with active load."
                elif diff < 0:
                    inv = "Unusually Low Draw: Potential power outage, trip breaker, or disconnected equipment."
                else:
                    inv = "Statistical Spike: Consumption significantly deviates from historical baseline."
            else:
                inv = "Normal consumption within standard operational distribution."
            investigations.append(inv)

        result_df["suggested_investigation"] = investigations
        return result_df

    def evaluate_single_reading(
        self,
        timestamp: str,
        consumption_kwh: float,
        occupancy: int,
        temperature: float,
        tariff_per_kwh: float = 0.15,
        operating_start: int = 8,
        operating_end: int = 18
    ) -> Dict[str, Any]:
        """Evaluate a single real-time or injected reading against the model."""
        dt = pd.to_datetime(timestamp)
        hour = dt.hour
        dow = dt.weekday()

        expected = self.hourly_means.get(hour, 1.5)
        diff_kwh = round(consumption_kwh - expected, 3)
        ratio = round(consumption_kwh / max(0.05, expected), 2)
        diff_cost = round(diff_kwh * tariff_per_kwh, 2)

        feat_df = pd.DataFrame([{
            "consumption_kwh": consumption_kwh,
            "hour": hour,
            "day_of_week": dow,
            "occupancy": occupancy,
            "temperature": temperature,
            "diff_from_hourly_mean": diff_kwh,
            "ratio_to_hourly_mean": ratio
        }])

        if self.model is None:
            if not self.load():
                self.hourly_means = {h: 2.0 for h in range(24)}

        if self.model is not None:
            pred = self.model.predict(feat_df[ANOMALY_FEATURE_COLS])[0]
            raw_score = float(self.model.decision_function(feat_df[ANOMALY_FEATURE_COLS])[0])
            score = round(float(np.clip((0.15 - raw_score) / 0.35, 0.0, 1.0)), 3)
            if abs(diff_kwh) <= 0.5 and 0.8 <= ratio <= 1.25:
                is_anomaly = 0
                score = min(score, 0.15)
            else:
                is_anomaly = 1 if pred == -1 else 0
        else:
            is_anomaly = 1 if (diff_kwh > 2.0 or ratio > 2.2) else 0
            score = 0.8 if is_anomaly else 0.1

        # Distinguish statistical anomaly from confirmed waste
        is_outside_hours = (hour < operating_start or hour >= operating_end)
        if is_anomaly:
            if diff_kwh > 0 and occupancy == 0 and is_outside_hours:
                waste_status = "CONFIRMED IDLE WASTE"
                explanation = f"Unusual off-hours consumption ({consumption_kwh:.2f} kWh vs {expected:.2f} kWh expected) with zero occupants. Devices left energized."
            elif diff_kwh > 0 and occupancy > 0:
                waste_status = "HIGH OPERATIONAL LOAD (NOT WASTE)"
                explanation = f"Consumption spike ({consumption_kwh:.2f} kWh) coincides with active occupancy ({occupancy} people). Legitimate high workload."
            elif diff_kwh < 0:
                waste_status = "ABNORMALLY LOW LOAD"
                explanation = f"Consumption dropped well below baseline ({consumption_kwh:.2f} kWh vs {expected:.2f} kWh expected). Inspect circuit breakers."
            else:
                waste_status = "SUSPECTED WASTE"
                explanation = f"Unusual energy draw ({consumption_kwh:.2f} kWh) exceeds baseline by {diff_kwh:.2f} kWh during unoccupied interval."
        else:
            waste_status = "NORMAL OPERATION"
            explanation = "Energy draw conforms to historical baseline distribution."

        return {
            "timestamp": timestamp,
            "measured_kwh": consumption_kwh,
            "expected_kwh": expected,
            "diff_kwh": diff_kwh,
            "diff_cost": diff_cost,
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": score,
            "waste_status": waste_status,
            "explanation": explanation,
            "recommended_action": (
                "Immediately inspect room switches, AC remote units, and workstation power strips."
                if is_anomaly and diff_kwh > 0 else "No intervention required."
            )
        }

    def save(self) -> str:
        path = self.get_model_path()
        payload = {
            "model": self.model,
            "hourly_means": self.hourly_means,
            "contamination": self.contamination,
            "location_id": self.location_id,
            "trained_at": self.trained_at,
            "total_trained_samples": self.total_trained_samples
        }
        joblib.dump(payload, path)
        return path

    def load(self) -> bool:
        path = self.get_model_path()
        if not os.path.exists(path):
            return False
        try:
            payload = joblib.load(path)
            self.model = payload["model"]
            self.hourly_means = payload.get("hourly_means", {})
            self.contamination = payload.get("contamination", self.contamination)
            self.trained_at = payload.get("trained_at")
            self.total_trained_samples = payload.get("total_trained_samples", 0)
            return True
        except Exception:
            return False

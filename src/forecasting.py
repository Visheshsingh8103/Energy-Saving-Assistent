"""
src/forecasting.py
Machine Learning Electricity Forecasting module for WattWise AI.
Uses Scikit-learn RandomForestRegressor with a strictly chronological train/test split,
fair benchmark against an hourly baseline model, and next-24-hour predictive pipeline.
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def build_forecasting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-specific calendar and physical features for electricity forecasting.
    Includes cyclical sine/cosine transforms for hour and day to preserve circular continuity.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical transforms (24h hour cycle, 7d week cycle)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # Fill defaults if missing
    if "temperature" not in df.columns:
        df["temperature"] = 24.0
    if "occupancy" not in df.columns:
        df["occupancy"] = 10

    return df


FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "temperature", "occupancy"
]


class HourlyBaselineModel:
    """
    A transparent baseline model that predicts the historical average consumption
    for the exact (hour, is_weekend) bucket computed strictly from training data.
    """
    def __init__(self):
        self.lookup_table: Dict[Tuple[int, int], float] = {}
        self.global_mean: float = 1.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "HourlyBaselineModel":
        data = X_train.copy()
        data["target"] = y_train.values
        grouped = data.groupby(["hour", "is_weekend"])["target"].mean()
        self.lookup_table = grouped.to_dict()
        self.global_mean = float(y_train.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X.iterrows():
            key = (int(row["hour"]), int(row["is_weekend"]))
            preds.append(self.lookup_table.get(key, self.global_mean))
        return np.array(preds)


class ElectricityForecaster:
    """
    End-to-end forecasting pipeline managing model training, chronological evaluation,
    model persistence, and future 24-hour horizon generation.
    """
    def __init__(self, location_id: str = "general"):
        self.location_id = location_id
        self.model: Optional[RandomForestRegressor] = None
        self.baseline_model: Optional[HourlyBaselineModel] = None
        self.feature_columns: List[str] = list(FEATURE_COLUMNS)
        self.metrics: Dict[str, Any] = {}
        self.trained_at: Optional[str] = None
        self.training_samples: int = 0
        self.test_samples: int = 0

    def get_model_path(self) -> str:
        os.makedirs(MODELS_DIR, exist_ok=True)
        return os.path.join(MODELS_DIR, f"forecaster_{self.location_id}.joblib")

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        test_fraction: float = 0.2,
        n_estimators: int = 80,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train RandomForestRegressor using strict chronological split to prevent future data leakage.
        Evaluates test metrics against the HourlyBaselineModel.
        """
        if len(df) < 100:
            raise ValueError(f"Insufficient historical observations for training ({len(df)} rows). Need at least 100.")

        # Ensure sorted by timestamp chronologically
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        df_feat = build_forecasting_features(df_sorted)

        X = df_feat[self.feature_columns]
        y = df_feat["consumption_kwh"]

        # Chronological split
        split_idx = int(len(df_feat) * (1.0 - test_fraction))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.training_samples = len(X_train)
        self.test_samples = len(X_test)

        # 1. Fit baseline model
        self.baseline_model = HourlyBaselineModel()
        self.baseline_model.fit(X_train, y_train)
        baseline_preds = self.baseline_model.predict(X_test)

        base_mae = float(mean_absolute_error(y_test, baseline_preds))
        base_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_preds)))

        # 2. Fit RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=14,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # 3. Evaluate on held-out test data
        test_preds = self.model.predict(X_test)
        rf_mae = float(mean_absolute_error(y_test, test_preds))
        rf_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))

        mae_improvement = round(((base_mae - rf_mae) / base_mae) * 100.0, 1) if base_mae > 0 else 0.0
        rmse_improvement = round(((base_rmse - rf_rmse) / base_rmse) * 100.0, 1) if base_rmse > 0 else 0.0

        # Feature importances
        importances = dict(zip(self.feature_columns, [round(float(v), 4) for v in self.model.feature_importances_]))

        self.trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metrics = {
            "trained_at": self.trained_at,
            "training_samples": self.training_samples,
            "test_samples": self.test_samples,
            "random_forest": {
                "mae": round(rf_mae, 4),
                "rmse": round(rf_rmse, 4)
            },
            "baseline_model": {
                "mae": round(base_mae, 4),
                "rmse": round(base_rmse, 4)
            },
            "improvement_pct": {
                "mae": mae_improvement,
                "rmse": rmse_improvement
            },
            "feature_importances": importances,
            "test_actual_sample": list(y_test.iloc[-72:].values), # last 72h test actuals
            "test_predicted_sample": [round(float(p), 3) for p in test_preds[-72:]],
            "test_timestamps_sample": [pd.to_datetime(t).strftime("%Y-%m-%d %H:%M") for t in df_sorted["timestamp"].iloc[split_idx:].iloc[-72:]]
        }

        # Persist model
        self.save()
        return self.metrics

    def predict_next_24h(
        self,
        latest_timestamp: Optional[datetime] = None,
        expected_operating_start: int = 8,
        expected_operating_end: int = 18,
        max_occupancy: int = 30
    ) -> pd.DataFrame:
        """
        Generate a 24-hour forward forecast using calendar features and reasonable diurnal profile.
        """
        if self.model is None:
            if not self.load():
                raise RuntimeError("Forecasting model has not been trained yet. Train the model first.")

        if latest_timestamp is None:
            latest_timestamp = datetime.now().replace(minute=0, second=0, microsecond=0)

        future_rows = []
        for i in range(1, 25):
            dt = latest_timestamp + timedelta(hours=i)
            hr = dt.hour
            dow = dt.weekday()
            is_weekend = 1 if dow >= 5 else 0

            # Diurnal temperature curve
            temp = round(22.0 + 6.0 * math.sin(2 * math.pi * (hr - 8) / 24.0), 1)

            # Scheduled occupancy assumption
            if expected_operating_start <= hr < expected_operating_end and not is_weekend:
                occ = int(round(max_occupancy * 0.75 * math.sin(math.pi * (hr - expected_operating_start) / (expected_operating_end - expected_operating_start))))
            else:
                occ = 0

            future_rows.append({
                "timestamp": dt,
                "hour": hr,
                "day_of_week": dow,
                "month": dt.month,
                "is_weekend": is_weekend,
                "temperature": temp,
                "occupancy": max(0, occ)
            })

        future_df = pd.DataFrame(future_rows)
        feat_df = build_forecasting_features(future_df)
        X_fut = feat_df[self.feature_columns]

        predicted_kwh = self.model.predict(X_fut)
        predicted_kwh = np.maximum(0.05, np.round(predicted_kwh, 3))

        # Reasonable prediction uncertainty interval based on model test RMSE
        rmse = self.metrics.get("random_forest", {}).get("rmse", 0.25)
        lower_bound = np.maximum(0.05, np.round(predicted_kwh - 1.96 * rmse, 3))
        upper_bound = np.round(predicted_kwh + 1.96 * rmse, 3)

        result_df = pd.DataFrame({
            "timestamp": future_df["timestamp"],
            "hour": future_df["hour"],
            "day_of_week": future_df["day_of_week"],
            "temperature": future_df["temperature"],
            "projected_occupancy": future_df["occupancy"],
            "predicted_kwh": predicted_kwh,
            "lower_bound_kwh": lower_bound,
            "upper_bound_kwh": upper_bound
        })
        return result_df

    def save(self) -> str:
        path = self.get_model_path()
        payload = {
            "model": self.model,
            "baseline_model": self.baseline_model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "location_id": self.location_id,
            "trained_at": self.trained_at
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
            self.baseline_model = payload.get("baseline_model")
            self.feature_columns = payload.get("feature_columns", FEATURE_COLUMNS)
            self.metrics = payload.get("metrics", {})
            self.trained_at = payload.get("trained_at")
            return True
        except Exception:
            return False

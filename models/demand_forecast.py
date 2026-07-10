"""
╔══════════════════════════════════════════════════════════════╗
║  🤖 ML ENGINEER AGENT — Demand Forecasting                  ║
║  Prophet-based hyper-local demand forecasting                ║
║  with weather and contextual signal regressors               ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, FORECASTS_DIR, FORECAST_HORIZON_HOURS,
    PIN_CODES, NUM_SKUS_FOR_FORECAST,
)


class DemandForecaster:
    """
    Hyper-local demand forecaster using Facebook Prophet.

    Predicts hourly SKU-level demand for specific pin codes
    with weather and contextual regressors.
    """

    def __init__(self):
        print("=" * 60)
        print("🤖 ML ENGINEER AGENT: Demand Forecasting System")
        print("=" * 60)

        self.demand_data = None
        self.weather_data = None
        self.models = {}  # Cache trained models
        self._load_data()

    def _load_data(self):
        """Load demand history and weather data."""
        demand_path = DATA_DIR / "demand_history.csv"
        weather_path = DATA_DIR / "weather_data.csv"

        if demand_path.exists():
            self.demand_data = pd.read_csv(demand_path, parse_dates=["timestamp"])
            print(f"   ✅ Demand data loaded: {len(self.demand_data):,} rows")
        else:
            print("   ❌ Demand history not found. Run generate_synthetic.py first.")
            raise FileNotFoundError(f"Demand data not found at {demand_path}")

        if weather_path.exists():
            self.weather_data = pd.read_csv(weather_path, parse_dates=["timestamp"])
            print(f"   ✅ Weather data loaded: {len(self.weather_data)} rows")

        # Get available SKUs and pin codes
        self.skus = sorted(self.demand_data["sku"].unique().tolist())
        self.pincodes = sorted(self.demand_data["pincode"].unique().tolist())
        print(f"   📊 SKUs: {len(self.skus)}, Pin codes: {len(self.pincodes)}")

    def prepare_prophet_data(self, sku, pincode):
        """
        Prepare data in Prophet format for a specific SKU-pincode combination.

        Prophet requires columns: ds (datetime), y (target value)
        Plus optional regressor columns.
        """
        mask = (self.demand_data["sku"] == sku) & (self.demand_data["pincode"] == pincode)
        subset = self.demand_data[mask].copy()

        if len(subset) == 0:
            raise ValueError(f"No data found for SKU={sku}, pincode={pincode}")

        df_prophet = pd.DataFrame({
            "ds": subset["timestamp"],
            "y": subset["net_demand"],
            "is_weekend": subset["is_weekend"].astype(float),
        })

        # Merge weather if available
        if self.weather_data is not None:
            weather = self.weather_data[["timestamp", "temperature", "precipitation"]].copy()
            weather.rename(columns={"timestamp": "ds"}, inplace=True)
            df_prophet = df_prophet.merge(weather, on="ds", how="left")
            df_prophet["temperature"].fillna(30.0, inplace=True)
            df_prophet["precipitation"].fillna(0.0, inplace=True)

        df_prophet.sort_values("ds", inplace=True)
        df_prophet.reset_index(drop=True, inplace=True)

        return df_prophet

    def train_forecast(self, sku, pincode, horizon_hours=FORECAST_HORIZON_HOURS):
        """
        Train a Prophet model and generate forecast.

        Args:
            sku: SKU identifier
            pincode: Pin code
            horizon_hours: Hours to forecast ahead

        Returns:
            (forecast_df, model, metrics_dict)
        """
        from prophet import Prophet

        model_key = f"{sku}_{pincode}"
        print(f"\n   🔮 Forecasting: {sku} @ {pincode}")

        # Prepare data
        df = self.prepare_prophet_data(sku, pincode)

        # Split: train on all but last 24h, validate on last 24h
        split_time = df["ds"].max() - timedelta(hours=horizon_hours)
        train = df[df["ds"] <= split_time]
        test = df[df["ds"] > split_time]

        # Initialize Prophet
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode="multiplicative",
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
        )

        # Add regressors
        model.add_regressor("is_weekend", mode="multiplicative")
        if "temperature" in df.columns:
            model.add_regressor("temperature", mode="additive")
            model.add_regressor("precipitation", mode="multiplicative")

        # Train
        model.fit(train)

        # Create future dataframe
        future = model.make_future_dataframe(periods=horizon_hours, freq="h")

        # Add regressor values for future
        future = future.merge(df[["ds", "is_weekend"]], on="ds", how="left")
        future["is_weekend"].fillna(
            future["ds"].dt.dayofweek.apply(lambda x: 1.0 if x >= 5 else 0.0),
            inplace=True,
        )

        if "temperature" in df.columns:
            future = future.merge(df[["ds", "temperature", "precipitation"]], on="ds", how="left")
            future["temperature"].fillna(30.0, inplace=True)
            future["precipitation"].fillna(0.0, inplace=True)

        # Predict
        forecast = model.predict(future)

        # Ensure non-negative predictions
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
        forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

        # Calculate metrics on test set
        metrics = {}
        if len(test) > 0:
            test_forecast = forecast[forecast["ds"].isin(test["ds"])].merge(
                test[["ds", "y"]], on="ds", how="inner"
            )
            if len(test_forecast) > 0:
                actual = test_forecast["y"].values
                predicted = test_forecast["yhat"].values

                # RMSE
                metrics["rmse"] = float(np.sqrt(np.mean((actual - predicted) ** 2)))

                # MAPE (avoid division by zero)
                nonzero_mask = actual > 0
                if nonzero_mask.sum() > 0:
                    metrics["mape"] = float(
                        np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])) * 100
                    )
                else:
                    metrics["mape"] = 0.0

                print(f"      RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.1f}%")

        # Cache model
        self.models[model_key] = model

        # Extract forecast for the future horizon
        future_mask = forecast["ds"] > split_time
        future_forecast = forecast[future_mask][
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()
        future_forecast.rename(columns={
            "ds": "timestamp",
            "yhat": "predicted_demand",
            "yhat_lower": "lower_bound",
            "yhat_upper": "upper_bound",
        }, inplace=True)
        future_forecast["sku"] = sku
        future_forecast["pincode"] = pincode

        return future_forecast, model, metrics

    def forecast_all(self, n_skus=5, n_pincodes=3):
        """
        Run forecasts for multiple SKU-pincode combinations.

        Returns:
            (all_forecasts_df, all_metrics)
        """
        print("\n🚀 Running Batch Forecasts")
        print("=" * 60)

        skus_to_forecast = self.skus[:n_skus]
        pincodes_to_forecast = self.pincodes[:n_pincodes]

        all_forecasts = []
        all_metrics = []

        for sku in skus_to_forecast:
            for pincode in pincodes_to_forecast:
                try:
                    forecast, _, metrics = self.train_forecast(sku, pincode)
                    all_forecasts.append(forecast)
                    all_metrics.append({
                        "sku": sku,
                        "pincode": pincode,
                        **metrics,
                    })
                except Exception as e:
                    print(f"      ⚠️  Failed for {sku}@{pincode}: {e}")

        if all_forecasts:
            combined = pd.concat(all_forecasts, ignore_index=True)
            metrics_df = pd.DataFrame(all_metrics)

            # Save
            combined.to_csv(FORECASTS_DIR / "forecasts.csv", index=False)
            metrics_df.to_csv(FORECASTS_DIR / "forecast_metrics.csv", index=False)

            print(f"\n   ✅ Forecasts generated: {len(all_forecasts)} SKU-pincode combos")
            print(f"   📁 Saved to {FORECASTS_DIR}")

            if len(metrics_df) > 0 and "mape" in metrics_df.columns:
                print(f"\n   📊 Aggregate Metrics:")
                print(f"      Avg MAPE: {metrics_df['mape'].mean():.1f}%")
                print(f"      Avg RMSE: {metrics_df['rmse'].mean():.2f}")

            return combined, metrics_df

        return pd.DataFrame(), pd.DataFrame()

    def get_demand_prediction(self, sku, pincode, hours=FORECAST_HORIZON_HOURS):
        """
        Get demand prediction for a specific SKU and pincode.
        Uses cached model if available, otherwise trains one.
        """
        model_key = f"{sku}_{pincode}"

        if model_key not in self.models:
            forecast, _, metrics = self.train_forecast(sku, pincode, hours)
            return forecast, metrics

        # Use cached model
        model = self.models[model_key]
        df = self.prepare_prophet_data(sku, pincode)

        future = model.make_future_dataframe(periods=hours, freq="h")
        future = future.merge(df[["ds", "is_weekend"]], on="ds", how="left")
        future["is_weekend"].fillna(
            future["ds"].dt.dayofweek.apply(lambda x: 1.0 if x >= 5 else 0.0),
            inplace=True,
        )
        if "temperature" in df.columns:
            future = future.merge(df[["ds", "temperature", "precipitation"]], on="ds", how="left")
            future["temperature"].fillna(30.0, inplace=True)
            future["precipitation"].fillna(0.0, inplace=True)

        forecast = model.predict(future)
        last_n = forecast.tail(hours)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        last_n.rename(columns={
            "ds": "timestamp",
            "yhat": "predicted_demand",
            "yhat_lower": "lower_bound",
            "yhat_upper": "upper_bound",
        }, inplace=True)
        last_n["predicted_demand"] = last_n["predicted_demand"].clip(lower=0)
        last_n["lower_bound"] = last_n["lower_bound"].clip(lower=0)
        last_n["upper_bound"] = last_n["upper_bound"].clip(lower=0)
        last_n["sku"] = sku
        last_n["pincode"] = pincode

        return last_n, {}

    def plot_forecast(self, sku, pincode):
        """Create an interactive plotly chart for a SKU-pincode forecast."""
        df = self.prepare_prophet_data(sku, pincode)
        forecast_df, _, metrics = self.train_forecast(sku, pincode)

        fig = make_subplots(rows=1, cols=1)

        # Historical data (last 7 days)
        last_7d = df[df["ds"] >= df["ds"].max() - timedelta(days=7)]
        fig.add_trace(
            go.Scatter(
                x=last_7d["ds"], y=last_7d["y"],
                mode="lines+markers",
                name="Historical Demand",
                line=dict(color="#3498db"),
                marker=dict(size=3),
            )
        )

        # Forecast
        fig.add_trace(
            go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df["predicted_demand"],
                mode="lines",
                name="Forecast",
                line=dict(color="#e74c3c", dash="dash"),
            )
        )

        # Confidence interval
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast_df["timestamp"], forecast_df["timestamp"][::-1]]),
                y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
                fill="toself",
                fillcolor="rgba(231, 76, 60, 0.15)",
                line=dict(color="rgba(231, 76, 60, 0)"),
                name="95% Confidence Interval",
            )
        )

        fig.update_layout(
            title=f"Demand Forecast: {sku} @ Pincode {pincode}",
            xaxis_title="Time",
            yaxis_title="Demand (units/hour)",
            template="plotly_dark",
            height=500,
            font=dict(family="Inter"),
        )

        return fig


def demo_forecasting():
    """Demo the forecasting system."""
    forecaster = DemandForecaster()
    forecasts, metrics = forecaster.forecast_all(n_skus=3, n_pincodes=2)

    print("\n" + "=" * 60)
    print("🎯 FORECAST DEMO RESULTS")
    print("=" * 60)

    if len(forecasts) > 0:
        print(f"\n📊 Sample Forecast (first SKU-pincode):")
        first_sku = forecasts["sku"].iloc[0]
        first_pin = forecasts["pincode"].iloc[0]
        sample = forecasts[(forecasts["sku"] == first_sku) & (forecasts["pincode"] == first_pin)]
        print(sample.head(10).to_string(index=False))

    return forecaster


if __name__ == "__main__":
    demo_forecasting()

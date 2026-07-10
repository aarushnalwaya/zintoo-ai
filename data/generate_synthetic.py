"""
╔══════════════════════════════════════════════════════════════╗
║  🔧 DATA ENGINEER AGENT — Synthetic Data Generator           ║
║  Creates warehouse inventory + demand history data           ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PIN_CODES, PIN_CODE_COORDS, WAREHOUSE_IDS, WAREHOUSE_PINCODE_MAP,
    REORDER_THRESHOLD, MAX_STOCK_PER_SKU, NUM_SKUS_FOR_FORECAST,
    HISTORY_DAYS, OPEN_METEO_URL, DATA_DIR, OUTPUTS_DIR,
    STYLES_CSV, DATASET_DIR,
)


def get_top_skus(n=NUM_SKUS_FOR_FORECAST):
    """Get top N product SKUs from the catalog for forecasting."""
    try:
        df = pd.read_csv(STYLES_CSV, on_bad_lines="skip")
        # Pick products from popular categories
        popular = df[df["masterCategory"].isin(["Apparel", "Footwear", "Accessories"])]
        skus = popular.sample(n=min(n, len(popular)), random_state=42)["id"].tolist()
        return [str(s) for s in skus]
    except Exception:
        # Fallback: generate synthetic SKU IDs
        return [f"SKU-{1000+i}" for i in range(n)]


def generate_warehouse_inventory(skus):
    """
    Generate micro-warehouse inventory data.

    Schema: product_id, sku, warehouse_id, pincode, current_stock, reorder_threshold
    """
    print("=" * 60)
    print("🔧 DATA ENGINEER AGENT: Generating Warehouse Inventory")
    print("=" * 60)

    rows = []
    for sku in skus:
        for wh_id in WAREHOUSE_IDS:
            pincode = WAREHOUSE_PINCODE_MAP[wh_id]
            current_stock = random.randint(5, MAX_STOCK_PER_SKU)
            rows.append({
                "product_id": sku,
                "sku": f"SKU-{sku}",
                "warehouse_id": wh_id,
                "pincode": pincode,
                "current_stock": current_stock,
                "reorder_threshold": REORDER_THRESHOLD,
                "max_capacity": MAX_STOCK_PER_SKU,
                "last_restocked": (
                    datetime.now() - timedelta(days=random.randint(1, 14))
                ).strftime("%Y-%m-%d"),
            })

    df = pd.DataFrame(rows)
    output_path = DATA_DIR / "warehouse_inventory.csv"
    df.to_csv(output_path, index=False)

    print(f"\n   ✅ Generated inventory for {len(WAREHOUSE_IDS)} warehouses × {len(skus)} SKUs")
    print(f"   📁 Saved to {output_path}")
    print(f"   📊 Total rows: {len(df)}")
    print(f"\n   Stock distribution:")
    print(f"      Min: {df['current_stock'].min()}")
    print(f"      Max: {df['current_stock'].max()}")
    print(f"      Mean: {df['current_stock'].mean():.1f}")

    return df


def fetch_weather_data(lat, lon, days_back=HISTORY_DAYS):
    """Fetch historical weather data from Open-Meteo API."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,weathercode",
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        weather_df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly.get("time", [])),
            "temperature": hourly.get("temperature_2m", []),
            "precipitation": hourly.get("precipitation", []),
            "weathercode": hourly.get("weathercode", []),
        })
        return weather_df

    except Exception as e:
        print(f"   ⚠️  Weather API failed ({e}), generating synthetic weather")
        return generate_synthetic_weather(days_back)


def generate_synthetic_weather(days_back):
    """Generate synthetic weather data as fallback."""
    timestamps = pd.date_range(
        end=datetime.now(),
        periods=days_back * 24,
        freq="h",
    )
    np.random.seed(42)
    n = len(timestamps)
    # Temperature: 25-38°C with daily cycle
    hour_of_day = timestamps.hour
    base_temp = 30 + 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    temp = base_temp + np.random.normal(0, 2, n)

    # Precipitation: mostly 0, occasional rain
    precip = np.zeros(n)
    rain_days = np.random.choice(range(0, n, 24), size=days_back // 5, replace=False)
    for rd in rain_days:
        duration = np.random.randint(2, 8)
        for h in range(duration):
            if rd + h < n:
                precip[rd + h] = np.random.exponential(5)

    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temp,
        "precipitation": precip,
        "weathercode": np.where(precip > 0, 61, 0),  # 61 = rain, 0 = clear
    })


def generate_demand_history(skus, weather_df=None):
    """
    Generate synthetic hourly demand data with realistic patterns.

    Patterns included:
    - Day-of-week seasonality (higher on weekends)
    - Hour-of-day patterns (peak at 10am, 6pm)
    - Weather correlation (rain → lower demand for some categories)
    - Random event spikes (festivals, sales)
    - Return rate (~15%)
    """
    print("\n" + "=" * 60)
    print("🔧 DATA ENGINEER AGENT: Generating Demand History")
    print("=" * 60)

    np.random.seed(42)
    random.seed(42)

    timestamps = pd.date_range(
        end=datetime.now().replace(minute=0, second=0, microsecond=0),
        periods=HISTORY_DAYS * 24,
        freq="h",
    )

    all_rows = []

    for sku in skus:
        # Base demand varies by "product popularity"
        base_demand = np.random.uniform(2, 8)

        for pincode in PIN_CODES:
            # Pincode multiplier (some areas buy more)
            pin_multiplier = np.random.uniform(0.5, 1.5)

            demands = []
            for ts in timestamps:
                # Hour-of-day pattern: peaks at 10am and 6pm
                hour = ts.hour
                hour_effect = (
                    0.3 + 0.7 * np.exp(-0.5 * ((hour - 10) / 3) ** 2) +
                    0.5 * np.exp(-0.5 * ((hour - 18) / 2) ** 2)
                )
                # Night suppression
                if hour < 6 or hour > 23:
                    hour_effect *= 0.1

                # Day-of-week: weekends 40% higher
                dow = ts.dayofweek
                dow_effect = 1.4 if dow >= 5 else 1.0

                # Weather effect (simplified)
                weather_effect = 1.0
                if weather_df is not None and len(weather_df) > 0:
                    weather_row = weather_df[weather_df["timestamp"] == ts]
                    if len(weather_row) > 0:
                        precip = weather_row.iloc[0]["precipitation"]
                        if precip > 5:
                            weather_effect = 0.6  # Heavy rain reduces demand
                        elif precip > 0:
                            weather_effect = 0.8

                # Random event spikes (10% chance on any day)
                event_effect = 1.0
                day_seed = hash(f"{sku}_{pincode}_{ts.date()}") % 100
                if day_seed < 5:  # 5% chance of a sale event
                    event_effect = 2.5
                elif day_seed < 10:  # 5% chance of festival
                    event_effect = 3.0

                # Compute demand
                demand = (
                    base_demand * pin_multiplier * hour_effect *
                    dow_effect * weather_effect * event_effect
                )

                # Add noise
                demand = max(0, int(np.random.poisson(max(0.1, demand))))
                demands.append(demand)

            # Generate returns (~15% of demand, delayed by 1-48 hours)
            returns = [0] * len(demands)
            for i, d in enumerate(demands):
                if d > 0:
                    n_returns = np.random.binomial(d, 0.15)
                    delay = np.random.randint(1, min(48, len(demands) - i))
                    if i + delay < len(returns):
                        returns[i + delay] += n_returns

            for ts, demand, ret in zip(timestamps, demands, returns):
                all_rows.append({
                    "timestamp": ts,
                    "sku": f"SKU-{sku}",
                    "pincode": pincode,
                    "demand": demand,
                    "returns": ret,
                    "net_demand": max(0, demand - ret),
                    "day_of_week": ts.strftime("%A"),
                    "hour": ts.hour,
                    "is_weekend": 1 if ts.dayofweek >= 5 else 0,
                })

    df = pd.DataFrame(all_rows)

    output_path = DATA_DIR / "demand_history.csv"
    df.to_csv(output_path, index=False)

    print(f"\n   ✅ Generated demand history for {len(skus)} SKUs × {len(PIN_CODES)} pin codes")
    print(f"   📅 Period: {HISTORY_DAYS} days ({len(timestamps)} hours)")
    print(f"   📁 Saved to {output_path}")
    print(f"   📊 Total rows: {len(df):,}")
    print(f"\n   Demand statistics:")
    print(f"      Total demand: {df['demand'].sum():,}")
    print(f"      Total returns: {df['returns'].sum():,}")
    print(f"      Avg hourly demand per SKU-pincode: {df['demand'].mean():.2f}")

    return df


def generate_all_synthetic_data():
    """Generate all synthetic data."""
    print("\n🚀 Starting synthetic data generation...\n")

    # Get SKUs
    skus = get_top_skus()
    print(f"📋 Using {len(skus)} SKUs for simulation")

    # Fetch or generate weather data (use first pincode's coords)
    lat, lon = PIN_CODE_COORDS[PIN_CODES[0]]
    print(f"\n🌤️  Fetching weather data for Mumbai ({lat}, {lon})...")
    weather_df = fetch_weather_data(lat, lon)
    weather_path = DATA_DIR / "weather_data.csv"
    weather_df.to_csv(weather_path, index=False)
    print(f"   ✅ Weather data: {len(weather_df)} hourly records → {weather_path}")

    # Generate warehouse inventory
    inventory_df = generate_warehouse_inventory(skus)

    # Generate demand history
    demand_df = generate_demand_history(skus, weather_df)

    print("\n" + "=" * 60)
    print("✅ ALL SYNTHETIC DATA GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n📁 Files created in {DATA_DIR}:")
    print(f"   • warehouse_inventory.csv  ({len(inventory_df)} rows)")
    print(f"   • demand_history.csv       ({len(demand_df):,} rows)")
    print(f"   • weather_data.csv         ({len(weather_df)} rows)")

    return skus, inventory_df, demand_df, weather_df


if __name__ == "__main__":
    generate_all_synthetic_data()

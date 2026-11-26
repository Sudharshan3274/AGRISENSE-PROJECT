"""
data_processor.py
------------------
Data pipeline for processing weather, rainfall, and agro-climatic indicators.
Computes multi-temporal rolling metrics, departures, dry spells, and feature vectors
for both rule-based risk evaluation and machine learning risk prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data") / "tamil_nadu_weather_2001_2024.csv"

def load_processed_weather(district=None, start_date=None, end_date=None):
    """
    Load historical weather data with calculated agro-climatic indicators.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Weather dataset not found at {DATA_PATH}. Run build_tamil_nadu_weather_dataset.py first.")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    if district:
        df = df[df["district"].str.lower() == district.lower()].copy()

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)].copy()

    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)].copy()

    return df

def compute_agro_features(df):
    """
    Computes rolling agro-meteorological features per district:
    - 7-day and 30-day rolling rainfall sums
    - 30-day cumulative normal rainfall and departure %
    - Consecutive dry days (rainfall < 2.5 mm, standard IMD dry day threshold)
    - Consecutive wet days (rainfall >= 2.5 mm)
    - Temperature anomalies and Heat Index proxy
    - Pest/humidity risk condition flag
    """
    df = df.sort_values(["district", "date"]).reset_index(drop=True)

    # Grouped rolling calculations
    grouped = df.groupby("district")

    df["rainfall_7d"] = grouped["rainfall"].transform(lambda s: s.rolling(7, min_periods=1).sum().round(2))
    df["rainfall_30d"] = grouped["rainfall"].transform(lambda s: s.rolling(30, min_periods=1).sum().round(2))
    df["normal_30d"] = grouped["normal_rainfall"].transform(lambda s: s.rolling(30, min_periods=1).sum().round(2))

    # 30-day departure (%)
    df["departure_30d"] = np.where(
        df["normal_30d"] > 0.1,
        np.round(((df["rainfall_30d"] - df["normal_30d"]) / df["normal_30d"]) * 100.0, 1),
        0.0
    )

    # Consecutive dry days calculation (rainfall < 2.5 mm)
    is_dry = (df["rainfall"] < 2.5).astype(int)
    
    # Calculate consecutive dry days
    dry_spell = []
    wet_spell = []
    
    current_district = None
    dry_count = 0
    wet_count = 0

    for dist, rain in zip(df["district"], df["rainfall"]):
        if dist != current_district:
            current_district = dist
            dry_count = 0
            wet_count = 0
            
        if rain < 2.5:
            dry_count += 1
            wet_count = 0
        else:
            wet_count += 1
            dry_count = 0
            
        dry_spell.append(dry_count)
        wet_spell.append(wet_count)

    df["consecutive_dry_days"] = dry_spell
    df["consecutive_wet_days"] = wet_spell

    # Simplified Heat Index approximation
    # HI = -8.784 + 1.611*T + 2.338*RH - 0.146*T*RH - 0.0123*T^2 - 0.0164*RH^2 + 0.0022*T^2*RH + 0.000725*T*RH^2
    # Simplified agro proxy:
    df["heat_index"] = np.round(df["tmax"] + 0.33 * (df["rh"] / 100.0 * 6.105 * np.exp((17.27 * df["tmax"]) / (237.7 + df["tmax"]))) - 4.0, 1)

    # Humidity-Temperature favorable pest zone (RH >= 80% and 24 <= T <= 32°C)
    df["pest_favorable"] = ((df["rh"] >= 80.0) & (df["tmean"] >= 24.0) & (df["tmean"] <= 32.0)).astype(int)

    return df

def get_district_list():
    """Returns sorted list of all available Tamil Nadu districts."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, usecols=["district"])
        return sorted(df["district"].unique().tolist())
    return []

if __name__ == "__main__":
    print("Testing data_processor.py...")
    sample_df = load_processed_weather(district="Chennai", start_date="2024-01-01", end_date="2024-12-31")
    sample_features = compute_agro_features(sample_df)
    print(f"Computed agro-features for Chennai 2024 ({len(sample_features)} rows).")
    print(sample_features[["date", "rainfall", "rainfall_7d", "rainfall_30d", "departure_30d", "consecutive_dry_days", "heat_index"]].tail(5))

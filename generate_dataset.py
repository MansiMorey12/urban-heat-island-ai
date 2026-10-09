"""
============================================================
generate_dataset.py  –  Synthetic UHI Dataset Generator
============================================================
Generates a realistic Urban Heat Island dataset that mirrors
the Kaggle "Urban Heat Island (UHI) Monitoring Dataset" schema.

  ► Run once: python generate_dataset.py
  ► Output  : data/uhi_dataset.csv

You can REPLACE data/uhi_dataset.csv with the real Kaggle CSV
and proceed directly to preprocessing.

NOTE: Temperature is generated from a REALISTIC per-city baseline
(each city's typical warm-season daytime temperature) plus daily
variation — not one flat random distribution shared by every city.
This means "Low / Medium / High" heat-level thresholds computed
later in preprocess.py actually reflect real-world temperature
differences between, say, London and Phoenix, instead of an
arbitrary synthetic average that doesn't resemble any real place.
============================================================
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_ROWS = 1000

# ── City metadata ────────────────────────────────────────────
# Approximate typical warm-season daytime temperature (°C) per city.
# These are rough, illustrative baselines (not official climate data)
# used only to make the synthetic dataset's spread realistic.
city_baseline_temp_c = {
    "Mumbai": 32, "Delhi": 34, "Bangalore": 28, "Chennai": 34, "Hyderabad": 33,
    "Kolkata": 33, "Pune": 31, "Ahmedabad": 36, "Jaipur": 35, "Lucknow": 34,
    "Nagpur": 36, "Bhopal": 32, "Indore": 32, "Chandigarh": 31, "Surat": 34,
    "New York": 24, "Los Angeles": 25, "Chicago": 22, "Houston": 30, "Phoenix": 39,
    "London": 18, "Tokyo": 25, "Beijing": 26, "Sydney": 24, "Dubai": 39,
}
cities = list(city_baseline_temp_c.keys())

land_use_types = ["Residential", "Commercial", "Industrial", "Park/Green Space", "Mixed Use"]

# ── Feature generation ───────────────────────────────────────
city_col = np.random.choice(cities, NUM_ROWS)
baseline_temp = np.array([city_baseline_temp_c[c] for c in city_col])
daily_variation = np.random.normal(0, 4, NUM_ROWS)  # day-to-day weather noise around each city's baseline

data = {
    "City": city_col,
    "Temperature_C": np.round(baseline_temp + daily_variation, 1),
    "Humidity_%": np.round(np.random.uniform(20, 95, NUM_ROWS), 1),
    "Wind_Speed_kmh": np.round(np.random.uniform(0, 40, NUM_ROWS), 1),
    "Green_Cover_%": np.round(np.random.uniform(2, 60, NUM_ROWS), 1),
    "Building_Density": np.round(np.random.uniform(10, 100, NUM_ROWS), 1),
    "Population_Density_per_km2": np.random.randint(500, 30000, NUM_ROWS),
    "Traffic_Density": np.round(np.random.uniform(10, 100, NUM_ROWS), 1),
    "Industrial_Proximity_km": np.round(np.random.uniform(0.5, 30, NUM_ROWS), 1),
    "Land_Use_Type": np.random.choice(land_use_types, NUM_ROWS),
    "Albedo": np.round(np.random.uniform(0.1, 0.6, NUM_ROWS), 2),
    "NDVI": np.round(np.random.uniform(0.05, 0.8, NUM_ROWS), 2),
    "Air_Quality_Index": np.random.randint(20, 400, NUM_ROWS),
    "Solar_Radiation_Wm2": np.round(np.random.uniform(100, 1000, NUM_ROWS), 1),
    "Rainfall_mm": np.round(np.random.uniform(0, 300, NUM_ROWS), 1),
    "Elevation_m": np.random.randint(0, 1500, NUM_ROWS),
}

# Make temperature correlate a bit with other features for realism
temp = data["Temperature_C"].copy()
temp += (data["Building_Density"] - 50) * 0.1    # more buildings → warmer
temp -= (data["Green_Cover_%"] - 30) * 0.15       # more green → cooler
temp -= (data["Wind_Speed_kmh"] - 20) * 0.08      # more wind → cooler
temp += (data["Traffic_Density"] - 50) * 0.05      # more traffic → warmer
data["Temperature_C"] = np.round(temp, 1)

# Original Heat_Island_Intensity (UHI effect in °C)
data["Heat_Island_Intensity_C"] = np.round(
    np.clip(data["Temperature_C"] - 28 + np.random.normal(0, 1.5, NUM_ROWS), 0, 15), 1
)

df = pd.DataFrame(data)

# ── Save ─────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "uhi_dataset.csv")
df.to_csv(output_path, index=False)

print(f"✅  Synthetic UHI dataset saved → {output_path}")
print(f"   Shape: {df.shape}")
print(f"\n📊 Temperature stats:\n{df['Temperature_C'].describe().round(1)}")
print(f"\n📊 Sample rows:\n{df.head()}")


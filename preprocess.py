"""
============================================================
preprocess.py  –  Data Preprocessing & Feature Engineering
============================================================
Reads the raw UHI dataset, creates balanced Heat_Level classes
using Temperature percentiles (Low / Medium / High), encodes
categorical features, scales numerics, and saves everything
for model training.

  ► Run: python preprocess.py
  ► Inputs : data/uhi_dataset.csv
  ► Outputs: data/X_train.csv, data/X_test.csv,
             data/y_train.csv, data/y_test.csv,
             models/scaler.pkl, models/label_encoder.pkl,
             models/feature_names.pkl
============================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")


def create_heat_level(df):
    """
    Create balanced Heat_Level classes using Temperature percentiles.
    ─────────────────────────────────────────────────────────────────
    • Low    : Temperature < 33rd percentile
    • Medium : 33rd ≤ Temperature < 67th percentile
    • High   : Temperature ≥ 67th percentile

    This avoids class imbalance that causes the model to predict
    only one class.
    """
    p33 = df["Temperature_C"].quantile(0.33)
    p67 = df["Temperature_C"].quantile(0.67)

    print(f"🌡️  Temperature thresholds:")
    print(f"   Low  : < {p33:.1f}°C")
    print(f"   Medium: {p33:.1f}°C – {p67:.1f}°C")
    print(f"   High : ≥ {p67:.1f}°C\n")

    conditions = [
        df["Temperature_C"] < p33,
        (df["Temperature_C"] >= p33) & (df["Temperature_C"] < p67),
        df["Temperature_C"] >= p67
    ]
    labels = ["Low", "Medium", "High"]
    df["Heat_Level"] = np.select(conditions, labels, default="Medium")
    return df


def preprocess_data():
    # ── 1. Load data ─────────────────────────────────────────
    csv_path = os.path.join("data", "uhi_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "Run 'python generate_dataset.py' first, or place "
            "the Kaggle CSV at data/uhi_dataset.csv."
        )
    
    df = pd.read_csv(csv_path)
    print(f"📂 Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # ── 2. Create target variable ────────────────────────────
    df = create_heat_level(df)

    class_dist = df["Heat_Level"].value_counts()
    print(f"📊 Heat Level distribution:\n{class_dist}\n")

    # ── 3. Select features ───────────────────────────────────
    # Drop non-feature columns
    drop_cols = ["City", "Heat_Level", "Heat_Island_Intensity_C"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Encode Land_Use_Type
    le_land = LabelEncoder()
    if "Land_Use_Type" in df.columns:
        df["Land_Use_Type_Encoded"] = le_land.fit_transform(df["Land_Use_Type"])
        drop_cols.append("Land_Use_Type")
    
    # Separate features and target
    y = df["Heat_Level"].copy()
    X = df.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    feature_names = list(X.columns)
    print(f"🔧 Features ({len(feature_names)}): {feature_names}\n")

    # ── 4. Encode target ─────────────────────────────────────
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    print(f"🏷️  Label mapping: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}\n")

    # ── 5. Train-test split ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"📐 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"   Train class distribution: {np.bincount(y_train)}")
    print(f"   Test  class distribution: {np.bincount(y_test)}\n")

    # ── 6. Scale features ────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index
    )

    # ── 7. Save everything ───────────────────────────────────
    os.makedirs("models", exist_ok=True)
    
    X_train_scaled.to_csv(os.path.join("data", "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join("data", "X_test.csv"), index=False)
    pd.DataFrame(y_train, columns=["Heat_Level"]).to_csv(
        os.path.join("data", "y_train.csv"), index=False
    )
    pd.DataFrame(y_test, columns=["Heat_Level"]).to_csv(
        os.path.join("data", "y_test.csv"), index=False
    )

    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    joblib.dump(le_target, os.path.join("models", "label_encoder.pkl"))
    joblib.dump(feature_names, os.path.join("models", "feature_names.pkl"))

    if "Land_Use_Type" in df.columns:
        joblib.dump(le_land, os.path.join("models", "land_use_encoder.pkl"))

    print("✅ Preprocessing complete! Saved:")
    print("   data/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print("   models/scaler.pkl, label_encoder.pkl, feature_names.pkl")


if __name__ == "__main__":
    preprocess_data()

# 🌍 AI-Driven Urban Heat Island Monitoring & Mitigation Planning Framework

A complete **beginner-friendly ML project** that predicts urban heat intensity levels and suggests actionable mitigation strategies using a Random Forest classifier and an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
urban-heat-island-ml/
├── data/                          # Dataset files
│   ├── uhi_dataset.csv           # Raw UHI dataset (generated or from Kaggle)
│   ├── X_train.csv               # Scaled training features
│   ├── X_test.csv                # Scaled test features
│   ├── y_train.csv               # Training labels
│   └── y_test.csv                # Test labels
├── models/                        # Saved model artifacts
│   ├── random_forest_model.pkl   # Trained Random Forest model
│   ├── scaler.pkl                # StandardScaler
│   ├── label_encoder.pkl         # Label encoder (High/Low/Medium)
│   ├── land_use_encoder.pkl      # Land use type encoder
│   └── feature_names.pkl         # Feature name list
├── outputs/                       # Training outputs
│   ├── confusion_matrix.png      # Confusion matrix heatmap
│   ├── feature_importance.png    # Feature importance bar chart
│   └── classification_report.txt # Precision/Recall/F1 report
├── generate_dataset.py           # Step 1: Generate synthetic UHI data
├── preprocess.py                 # Step 2: Preprocess & create balanced classes
├── train_model.py                # Step 3: Train Random Forest model
├── app.py                        # Step 4: Streamlit dashboard
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset (or use the real Kaggle CSV)
```bash
python generate_dataset.py
```
> **Tip:** You can replace `data/uhi_dataset.csv` with the real dataset from
> [Kaggle](https://www.kaggle.com/datasets/atharvasoundankar/urban-heat-island-uhi-monitoring-dataset)

### 3. Preprocess data
```bash
python preprocess.py
```

### 4. Train the model
```bash
python train_model.py
```

### 5. Launch the dashboard
```bash
python -m streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🧠 How It Works

### Class Imbalance Fix
Instead of using the raw (possibly imbalanced) labels, we create **3 balanced classes** using Temperature percentiles:

| Class    | Rule                          |
|----------|-------------------------------|
| **Low**    | Temperature < 33rd percentile |
| **Medium** | 33rd – 67th percentile       |
| **High**   | ≥ 67th percentile            |

This ensures the model learns to predict **all three classes** rather than defaulting to the majority class.

### Model
- **Algorithm:** Random Forest (200 trees, max_depth=15)
- **Class balancing:** `class_weight='balanced'` as additional safeguard
- **Features:** 15 environmental & urban parameters

### Dashboard Features
- 📍 **Auto-Fill From a Real Location** — type a city/place and the app fetches live temperature,
  humidity, wind speed, rainfall, solar radiation, elevation, and air quality (via the free
  [Open-Meteo](https://open-meteo.com) APIs — no key required), and estimates building density,
  green cover, land use, traffic density, NDVI, albedo, and population density from real
  [OpenStreetMap](https://www.openstreetmap.org) data near that location. Every value is still
  adjustable afterward.
- 🌡️ **Heat Level Prediction** with confidence gauges
- 📋 **Prediction Reasons** showing top contributing factors
- 🛡️ **Mitigation Strategies** — trees, cool roofs, green belts, rainwater harvesting
- 📊 **Feature Importance** visualization

> **Note on the "estimated" fields:** free live satellite feeds for vegetation index, building
> density, and population density don't exist without a paid API key or Earth-observation account.
> Those fields are computed from OpenStreetMap counts near the entered location using simple,
> documented formulas (see `derive_urban_estimates()` in `app.py`) rather than left for the user
> to guess.

---

## 📊 Features Used

| Feature | Description |
|---------|-------------|
| Temperature_C | Ambient temperature (°C) |
| Humidity_% | Relative humidity (%) |
| Wind_Speed_kmh | Wind speed (km/h) |
| Green_Cover_% | Percentage of green/vegetation cover |
| Building_Density | Building density index |
| Population_Density_per_km2 | People per km² |
| Traffic_Density | Traffic density index |
| Industrial_Proximity_km | Distance to nearest industrial zone |
| Albedo | Surface reflectivity (0-1) |
| NDVI | Normalized Difference Vegetation Index |
| Air_Quality_Index | AQI value |
| Solar_Radiation_Wm2 | Incoming solar radiation (W/m²) |
| Rainfall_mm | Rainfall amount (mm) |
| Elevation_m | Altitude above sea level |
| Land_Use_Type_Encoded | Encoded land use category |

---

## 📜 License

This project uses the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) licensed UHI dataset from Kaggle.

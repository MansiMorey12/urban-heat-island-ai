"""
============================================================
app.py  –  Streamlit Dashboard
============================================================
AI-Driven Urban Heat Island Monitoring & Mitigation Planning

Navigation:
  🏠 Home        → Overview, stats, project info
  🔮 Predict     → Enter values, get prediction + mitigation
  📊 Analytics   → Model performance & feature importance
  ℹ️ About       → Project documentation

  ► Run: python -m streamlit run app.py
============================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
import requests
from math import radians, sin, cos, sqrt, atan2
from explainable_mitigation import get_shap_explainer, explain_and_select_mitigation

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Urban Heat Island Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── ROOT SIZE BOOST — scales everything up ─────────── */
    html {
        font-size: 20px !important;
    }

    /* ── Main background ─────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e94560 !important;
    }
    section[data-testid="stSidebar"] label span {
        font-size: 1.15rem !important;
    }

    /* ── Glass cards ─────────────────────────────────────── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 32px 36px;
        margin: 12px 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(233, 69, 96, 0.12);
    }

    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(233, 69, 96, 0.2);
    }
    .metric-card h2 { margin: 0; font-size: 3.2rem; }
    .metric-card p  { margin: 6px 0 0; opacity: 0.7; font-size: 1.3rem; }

    /* ── Hero section ────────────────────────────────────── */
    .hero-section {
        text-align: center;
        padding: 60px 20px 35px;
    }
    .hero-title {
        font-size: 6.5rem;      /* Increased from 4.2rem */
        font-weight: 900;
        background: linear-gradient(135deg, #00d2ff 0%, #f9a825 50%, #ff1744 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 20px;
        line-height: 1.15;
        text-align: center;
    }
    .hero-subtitle {
        font-size: 2rem;
        opacity: 0.7;
        max-width: 900px;
        margin: 0 auto;
        line-height: 1.7;
    }
    /* ── Feature cards grid ──────────────────────────────── */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 24px;
        margin: 35px 0;
    }
    .feature-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 32px 28px;
        text-align: center;
        transition: all 0.35s ease;
    }
    .feature-card:hover {
        background: rgba(255,255,255,0.06);
        transform: translateY(-6px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .feature-card .icon { font-size: 3.8rem; margin-bottom: 16px; }
    .feature-card h3 {
        font-size: 1.6rem; font-weight: 700;
        margin: 0 0 12px; color: #f0f0f0;
    }
    .feature-card p {
        font-size: 1.25rem; opacity: 0.55; margin: 0; line-height: 1.7;
    }

    /* ── Stats row ───────────────────────────────────────── */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 60px;
        margin: 50px 0;
    }
    .stat-item { text-align: center; }
    .stat-value {
        font-size: 3.8rem; font-weight: 800;
        background: linear-gradient(135deg, #00d2ff, #e94560);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label { font-size: 1.2rem; opacity: 0.45; margin-top: 8px; letter-spacing: 1px; }

    /* ── Heat level badges ───────────────────────────────── */
    .heat-low    { color: #00d2ff; text-shadow: 0 0 30px rgba(0,210,255,0.5); }
    .heat-medium { color: #f9a825; text-shadow: 0 0 30px rgba(249,168,37,0.5); }
    .heat-high   { color: #ff1744; text-shadow: 0 0 30px rgba(255,23,68,0.5); }

    /* ── Reason / mitigation boxes ───────────────────────── */
    .reason-box {
        background: rgba(233, 69, 96, 0.06);
        border-left: 5px solid #e94560;
        border-radius: 0 14px 14px 0;
        padding: 22px 28px;
        margin: 14px 0;
        font-size: 1.3rem;
    }
    .mitigation-box {
        background: rgba(46,204,113,0.06);
        border-left: 5px solid #2ecc71;
        border-radius: 0 14px 14px 0;
        padding: 22px 28px;
        margin: 14px 0;
        transition: transform 0.2s ease;
        font-size: 1.3rem;
    }
    .mitigation-box:hover { transform: translateX(6px); }
    .mitigation-box h4 { color: #2ecc71; margin: 0 0 10px; font-size: 1.45rem; }
    .reason-box h4     { color: #e94560; margin: 0 0 10px; font-size: 1.45rem; }

    /* ── Section headers ─────────────────────────────────── */
    .section-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 40px 0 18px;
        padding-bottom: 14px;
        border-bottom: 2px solid rgba(233,69,96,0.25);
    }

    /* ── Nav buttons in sidebar ──────────────────────────── */
    .nav-item {
        display: block;
        padding: 14px 20px;
        margin: 6px 0;
        border-radius: 12px;
        color: rgba(255,255,255,0.7);
        text-decoration: none;
        font-size: 1.3rem;
        font-weight: 500;
        transition: all 0.25s ease;
        cursor: pointer;
    }
    .nav-item:hover {
        background: rgba(233,69,96,0.12);
        color: #fff;
    }
    .nav-active {
        background: linear-gradient(135deg, rgba(233,69,96,0.15), rgba(0,210,255,0.1));
        color: #fff !important;
        border-left: 3px solid #e94560;
    }

    /* ── CTA button ──────────────────────────────────────── */
    .cta-container { text-align: center; margin: 40px 0; }

    /* ── Divider ─────────────────────────────────────────── */
    .glow-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(233,69,96,0.4), transparent);
        border: none;
        margin: 35px 0;
    }

    /* ── Prediction result card ──────────────────────────── */
    .result-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 24px;
        padding: 50px;
        text-align: center;
        margin: 24px auto;
        max-width: 550px;
    }
    .result-label {
        font-size: 1.4rem;
        opacity: 0.5;
        text-transform: uppercase;
        letter-spacing: 4px;
        margin-bottom: 16px;
    }
    .result-value {
        font-size: 4.5rem;
        font-weight: 800;
        margin: 12px 0;
    }
    .result-confidence {
        font-size: 1.5rem;
        opacity: 0.6;
    }

    /* Hide default Streamlit stuff */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}

    /* ── Slider styling ──────────────────────────────────── */
        .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #e94560 !important;
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #00d2ff, #f9a825, #ff1744) !important;
    }
    /* ── Input form styling ──────────────────────────────── */
    .input-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 32px;
        margin: 18px 0;
    }
    .input-section h3 {
        color: #e94560;
        margin-top: 0;
        font-size: 1.6rem;
    }

    /* ── GLOBAL TEXT SIZE OVERRIDES ───────────────────────── */
    /* Force all Streamlit text elements to be large */
    .stApp p, .stApp li, .stApp span, .stApp label,
    .stApp div, .stApp td, .stApp th {
        font-size: 1.25rem !important;
    }
    .stApp h1 { font-size: 2.8rem !important; }
    .stApp h2 { font-size: 2.2rem !important; }
    .stApp h3 { font-size: 1.8rem !important; }
    .stApp h4 { font-size: 1.5rem !important; }

    /* Sidebar radio / nav labels */
    .stApp .stRadio label span,
    section[data-testid="stSidebar"] .stRadio label span {
        font-size: 1.3rem !important;
    }
    /* Slider labels */
    .stApp .stSlider label p,
    .stApp .stSlider label span {
        font-size: 1.2rem !important;
    }
    /* Slider value display */
    .stApp .stSlider [data-testid="stThumbValue"] {
        font-size: 1.15rem !important;
    }
    /* Selectbox labels */
    .stApp .stSelectbox label p,
    .stApp .stSelectbox label span {
        font-size: 1.2rem !important;
    }
    /* Selectbox value */
    .stApp .stSelectbox [data-testid="stMarkdownContainer"],
    .stApp .stSelectbox div[role="combobox"] {
        font-size: 1.2rem !important;
    }
    /* Expander */
    .stApp .stExpander summary span {
        font-size: 1.35rem !important;
    }
    /* Buttons */
    .stApp button {
        font-size: 1.3rem !important;
    }
    .stApp button[kind="primary"] {
        font-size: 1.4rem !important;
        padding: 14px 28px !important;
    }
    /* Data frames / tables */
    .stApp .stDataFrame td, .stApp .stDataFrame th {
        font-size: 1.15rem !important;
    }
    /* Code blocks */
    .stApp pre, .stApp code {
        font-size: 1.1rem !important;
    }
    /* Glass card overrides */
    .glass-card p {
        font-size: 1.3rem !important;
    }
    .glass-card li {
        font-size: 1.25rem !important;
    }
    .glass-card h3 {
        font-size: 1.7rem !important;
    }
    .glass-card h4 {
        font-size: 1.45rem !important;
    }
    /* Info, warning, error boxes */
    .stApp .stAlert p {
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  LOAD MODEL & ARTIFACTS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join("models", "random_forest_model.pkl"))
    scaler = joblib.load(os.path.join("models", "scaler.pkl"))
    le = joblib.load(os.path.join("models", "label_encoder.pkl"))
    feature_names = joblib.load(os.path.join("models", "feature_names.pkl"))
    land_enc = None
    land_enc_path = os.path.join("models", "land_use_encoder.pkl")
    if os.path.exists(land_enc_path):
        land_enc = joblib.load(land_enc_path)
    return model, scaler, le, feature_names, land_enc

try:
    model, scaler, le, feature_names, land_enc = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ══════════════════════════════════════════════════════════════
#  EXPLAINABLE AI: SHAP EXPLAINER (for prediction-specific,
#  model-driven mitigation selection — see explainable_mitigation.py)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_shap_explainer(_model):
    return get_shap_explainer(_model)

if model_loaded:
    try:
        shap_explainer = load_shap_explainer(model)
        shap_loaded = True
    except Exception as e:
        shap_loaded = False
        shap_error = str(e)
else:
    shap_loaded = False


# ══════════════════════════════════════════════════════════════
#  AUTO-FETCH: LIVE LOCATION DATA
#  Uses free, no-key APIs (Open-Meteo + OpenStreetMap Overpass)
#  so the user only has to type a place name instead of guessing
#  15 environmental values by hand.
# ══════════════════════════════════════════════════════════════

def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in km."""
    r = 6371.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


# Public APIs (especially Overpass) commonly reject requests that use the
# default python-requests User-Agent, treating it as an unidentified bot.
# Sending a descriptive one is standard practice and fixes that.
_HEADERS = {"User-Agent": "UrbanHeatIslandStudentProject/1.0 (Streamlit app; educational use)"}


@st.cache_data(ttl=600, show_spinner=False)
def geocode_place(place_name):
    """
    Turn a typed place name into lat/lon. Tries Nominatim (OpenStreetMap's
    geocoder) first since it understands neighborhoods, streets and
    landmarks — not just cities. Falls back to Open-Meteo's geocoder
    (cities/towns only) if Nominatim doesn't respond.
    """
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place_name, "format": "json", "limit": 1, "addressdetails": 1},
            headers=_HEADERS, timeout=10,
        )
        r.raise_for_status()
        results = r.json()
        if results:
            res = results[0]
            return {
                "lat": float(res["lat"]),
                "lon": float(res["lon"]),
                "label": res.get("display_name", place_name).split(",")[0] + ", " +
                         res.get("address", {}).get("country", ""),
            }
    except Exception:
        pass

    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place_name, "count": 1}, headers=_HEADERS, timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results")
        if results:
            res = results[0]
            return {
                "lat": res["latitude"],
                "lon": res["longitude"],
                "label": f"{res.get('name', place_name)}, {res.get('country', '')}".strip(", "),
            }
    except Exception:
        pass

    return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather(lat, lon):
    """Real, live values: temperature, humidity, wind, rainfall, sunlight, elevation, AQI."""
    result = {
        "temperature": None, "humidity": None, "wind_speed": None,
        "rainfall": None, "solar_radiation": None, "elevation": None, "aqi": None,
    }
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation",
                "daily": "precipitation_sum",
                "timezone": "auto",
            },
            headers=_HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        w = r.json()
        current = w.get("current", {})
        daily = w.get("daily", {})
        result["temperature"] = current.get("temperature_2m")
        result["humidity"] = current.get("relative_humidity_2m")
        result["wind_speed"] = current.get("wind_speed_10m")
        result["solar_radiation"] = current.get("shortwave_radiation")
        precip_list = daily.get("precipitation_sum") or []
        result["rainfall"] = precip_list[0] if precip_list else 0.0
    except Exception:
        pass

    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": lat, "longitude": lon}, headers=_HEADERS, timeout=10,
        )
        r.raise_for_status()
        elevs = r.json().get("elevation") or []
        result["elevation"] = elevs[0] if elevs else None
    except Exception:
        pass

    try:
        r = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude": lat, "longitude": lon, "current": "us_aqi"}, headers=_HEADERS, timeout=10,
        )
        r.raise_for_status()
        result["aqi"] = r.json().get("current", {}).get("us_aqi")
    except Exception:
        pass

    return result


_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def _overpass_post(query, errors):
    """Try each Overpass mirror in turn. Returns the parsed JSON, or None if all fail."""
    for url in _OVERPASS_MIRRORS:
        try:
            r = requests.post(url, data={"data": query}, headers=_HEADERS, timeout=25)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            errors.append(f"{url.split('/')[2]}: {e}")
    return None


def fetch_osm_features(lat, lon, radius_m=900, force_refresh=False):
    """
    Approximate urban structure from free OpenStreetMap data:
    building count, green-space feature count, major-road count,
    and distance to the nearest industrial zone.

    All four are fetched in a SINGLE Overpass request (one round trip
    instead of four) to avoid tripping rate limits. 'ok' is True if we
    got any real data back at all — partial results are still used
    rather than thrown away.

    Uses a manual session-state cache instead of @st.cache_data, because
    st.cache_data would also cache FAILED attempts for its whole TTL —
    meaning a retry a minute later would just replay the same failure.
    Only successful results get cached; failures are never cached, so
    clicking retry always tries fresh.
    """
    cache_key = f"_osm_cache_{round(lat, 3)}_{round(lon, 3)}_{radius_m}"
    if not force_refresh and cache_key in st.session_state:
        return st.session_state[cache_key]

    errors = []
    query = (
        f'[out:json][timeout:25];'
        f'way["building"](around:{radius_m},{lat},{lon})->.b;'
        f'.b out count;'
        f'('
        f'way["leisure"="park"](around:{radius_m},{lat},{lon});'
        f'way["landuse"="forest"](around:{radius_m},{lat},{lon});'
        f'way["landuse"="grass"](around:{radius_m},{lat},{lon});'
        f'way["natural"="wood"](around:{radius_m},{lat},{lon});'
        f')->.g;'
        f'.g out count;'
        f'way["highway"~"motorway|trunk|primary|secondary"](around:{radius_m},{lat},{lon})->.r;'
        f'.r out count;'
        f'way["landuse"="industrial"](around:5000,{lat},{lon})->.i;'
        f'.i out center 5;'
    )
    data = _overpass_post(query, errors)

    building_count = green_count = road_count = 0
    industrial_distance_km = 30.0  # default: nothing found nearby = "very far"
    ok = data is not None

    if data is not None:
        counts = [
            int(el.get("tags", {}).get("ways", 0))
            for el in data.get("elements", []) if el.get("type") == "count"
        ]
        if len(counts) >= 1:
            building_count = counts[0]
        if len(counts) >= 2:
            green_count = counts[1]
        if len(counts) >= 3:
            road_count = counts[2]

        dists = [
            _haversine_km(lat, lon, el["center"]["lat"], el["center"]["lon"])
            for el in data.get("elements", []) if el.get("center")
        ]
        if dists:
            industrial_distance_km = min(dists)

    result = {
        "building_count": building_count,
        "green_count": green_count,
        "road_count": road_count,
        "industrial_distance_km": round(industrial_distance_km, 1),
        "ok": ok,
        "errors": errors,
    }
    if ok:
        st.session_state[cache_key] = result
    return result


def derive_urban_estimates(osm):
    """
    Convert raw OSM counts into estimates for the fields that have no
    free live API (building density, green cover, land use, traffic,
    NDVI, albedo, population density). Each formula is a simple,
    explainable heuristic — not a guess, but not satellite-grade either.
    """
    building_count = osm["building_count"]
    green_count = osm["green_count"]
    road_count = osm["road_count"]
    industrial_km = osm["industrial_distance_km"]

    building_density = float(np.clip(10 + building_count * 0.12, 10, 100))
    green_cover = float(np.clip(2 + green_count * 3.5, 2, 60))
    traffic_density = float(np.clip(10 + road_count * 4, 10, 100))

    if green_count > building_count and green_count > road_count:
        land_use = "Park/Green Space"
    elif industrial_km < 1.5:
        land_use = "Industrial"
    elif building_count > 30 and road_count > building_count * 0.3:
        land_use = "Commercial"
    elif building_count > 30:
        land_use = "Residential"
    else:
        land_use = "Mixed Use"

    ndvi = float(np.clip(0.05 + (green_cover / 60) * 0.75, 0.05, 0.8))
    albedo = float(np.clip(0.6 - (building_density / 100) * 0.45, 0.1, 0.6))
    population_density = int(np.clip(500 + building_density * 280, 500, 30000))

    return {
        "building_density": building_density,
        "green_cover": green_cover,
        "traffic_density": traffic_density,
        "land_use": land_use,
        "ndvi": ndvi,
        "albedo": albedo,
        "population_density": population_density,
        "industrial_distance_km": industrial_km,
    }


def closest_label(value, mapping):
    """Given a numeric value and a {label: number} map, return the nearest label."""
    return min(mapping.items(), key=lambda kv: abs(kv[1] - value))[0]


# ── Slider <-> real-world-value lookup tables (shared by the manual
#    sliders below and the auto-fill logic above) ────────────────
sunshine_map = {"Cloudy / Overcast": 150, "Partly Sunny": 350, "Sunny": 550, "Very Sunny": 800, "Blazing Sun": 1100}
elevation_map = {"Sea Level / Low": 30, "Slightly Elevated": 200, "Moderate Height": 500, "Hilly": 900, "High Altitude": 1500}
green_map = {"Almost None": 5, "Very Little": 12, "Some Parks": 25, "Good Amount": 40, "Lots of Trees & Parks": 60}
building_map = {"Very Sparse": 15, "Spread Out": 30, "Moderate": 50, "Dense / Packed": 75, "Extremely Crowded": 95}
pop_map = {"Very Few": 1000, "Low Density": 4000, "Medium Density": 8000, "Crowded": 18000, "Very Crowded City": 30000}
traffic_map = {"Almost No Traffic": 10, "Light Traffic": 25, "Moderate Traffic": 50, "Heavy Traffic": 75, "Constant Jams": 95}
factory_map = {"Right Next Door": 1, "Nearby (1-3 km)": 3, "Moderate Distance": 10, "Far Away": 20, "Very Far / None": 30}
albedo_map = {"Very Dark (Black tar)": 0.10, "Dark (Grey)": 0.20, "Medium": 0.35, "Light (White/Beige)": 0.55, "Very Light / Reflective": 0.75}
ndvi_map = {"Bare / Dead": 0.05, "Dry & Brown": 0.15, "Some Greenery": 0.35, "Healthy & Green": 0.55, "Lush & Dense": 0.80}
aqi_map = {"Clean & Fresh": 30, "Good": 60, "Moderate": 120, "Unhealthy": 250, "Very Polluted": 420}
land_use_options = {
    "🏠 Houses & Apartments": "Residential",
    "🏬 Shops & Offices": "Commercial",
    "🏭 Factories & Warehouses": "Industrial",
    "🌳 Parks & Gardens": "Park/Green Space",
    "🏘️ Mix of Everything": "Mixed Use",
}
reverse_land_use_options = {v: k for k, v in land_use_options.items()}


# ══════════════════════════════════════════════════════════════
#  MITIGATION STRATEGIES
#  (Fallback / general library — used only if SHAP-based,
#  prediction-specific mitigation cannot be computed for some
#  reason, e.g. shap_loaded is False. See explainable_mitigation.py
#  for the primary, explainable-AI-driven selection logic.)
# ══════════════════════════════════════════════════════════════
MITIGATION = {
    "Low": [
        {
            "icon": "🌳",
            "title": "Maintain Green Cover",
            "desc": "Continue maintaining existing urban forests and parks. Current green cover is adequate — focus on preservation."
        },
        {
            "icon": "💧",
            "title": "Rainwater Harvesting",
            "desc": "Install basic rainwater collection systems to maintain soil moisture and support vegetation growth."
        },
    ],
    "Medium": [
        {
            "icon": "🌳",
            "title": "Expand Urban Tree Canopy",
            "desc": "Plant medium-to-large native shade trees along streets and in parking lots. Target 25-40% canopy cover to reduce ambient temperature by 2-4°C."
        },
        {
            "icon": "🏠",
            "title": "Cool Roof Initiative",
            "desc": "Apply high-albedo reflective coatings (albedo > 0.6) on commercial and residential rooftops. Cool roofs can reduce surface temperature by up to 30°C."
        },
        {
            "icon": "💧",
            "title": "Rainwater Harvesting Systems",
            "desc": "Deploy rooftop rainwater harvesting with storage tanks. Use collected water for landscape irrigation to sustain green cover during dry spells."
        },
        {
            "icon": "🛣️",
            "title": "Permeable Pavements",
            "desc": "Replace conventional pavements with permeable materials to reduce heat absorption and improve groundwater recharge."
        },
    ],
    "High": [
        {
            "icon": "🌳",
            "title": "Aggressive Afforestation & Green Belts",
            "desc": "Establish wide green belts (50-100m) around industrial and commercial zones. Plant at least 100 trees per hectare with native species for maximum cooling."
        },
        {
            "icon": "🏠",
            "title": "Mandatory Cool Roofs & Green Roofs",
            "desc": "Enforce building codes requiring cool roofs (albedo > 0.7) on all new construction. Retrofit existing buildings with green roof systems for insulation and evaporative cooling."
        },
        {
            "icon": "💧",
            "title": "Integrated Rainwater Harvesting",
            "desc": "Implement large-scale rainwater harvesting at community level. Build underground cisterns and connect to drip irrigation for public green spaces."
        },
        {
            "icon": "🌿",
            "title": "Urban Green Corridors",
            "desc": "Create interconnected green corridors linking parks, gardens, and water bodies. These act as ventilation channels to flush out trapped heat."
        },
        {
            "icon": "🚗",
            "title": "Traffic & Emission Controls",
            "desc": "Implement congestion pricing, promote EV adoption, and create car-free zones in high-density areas to reduce anthropogenic heat."
        },
        {
            "icon": "🏗️",
            "title": "Cool Pavement & Infrastructure",
            "desc": "Deploy cool pavement technology with reflective aggregate. Retrofit bus stops, walkways, and public spaces with shade structures and misting systems."
        },
    ]
}


def get_prediction_reasons(input_df, prediction, feature_importances, feature_names):
    """Generate human-readable reasons for the prediction."""
    reasons = []
    values = input_df.values[0]

    feature_info = {
        "Temperature_C": ("Temperature", "°C", 35, "higher temperatures intensify urban heat islands"),
        "Humidity_%": ("Humidity", "%", 60, "high humidity traps heat and reduces evaporative cooling"),
        "Wind_Speed_kmh": ("Wind Speed", " km/h", 15, "low wind speed reduces natural heat dissipation"),
        "Green_Cover_%": ("Green Cover", "%", 20, "insufficient green cover reduces natural cooling"),
        "Building_Density": ("Building Density", "", 60, "dense buildings trap and re-radiate heat"),
        "Population_Density_per_km2": ("Population Density", " /km²", 10000, "high population increases anthropogenic heat output"),
        "Traffic_Density": ("Traffic Density", "", 60, "heavy traffic generates significant waste heat"),
        "Industrial_Proximity_km": ("Industrial Proximity", " km", 5, "proximity to industrial zones increases ambient temperature"),
        "Albedo": ("Surface Albedo", "", 0.3, "low albedo means surfaces absorb more solar heat"),
        "NDVI": ("Vegetation Index (NDVI)", "", 0.3, "low NDVI indicates sparse vegetation and less cooling"),
        "Air_Quality_Index": ("Air Quality Index", "", 150, "poor air quality correlates with heat retention"),
        "Solar_Radiation_Wm2": ("Solar Radiation", " W/m²", 600, "high solar radiation directly heats urban surfaces"),
        "Rainfall_mm": ("Rainfall", " mm", 50, "low rainfall reduces evaporative cooling capacity"),
        "Elevation_m": ("Elevation", " m", 300, "lower elevation often correlates with warmer conditions"),
    }

    sorted_idx = np.argsort(feature_importances)[::-1]

    for idx in sorted_idx[:5]:
        fname = feature_names[idx]
        fval = values[idx]
        if fname in feature_info:
            readable, unit, threshold, explanation = feature_info[fname]
            if fname in ["Wind_Speed_kmh", "Green_Cover_%", "NDVI", "Albedo", "Rainfall_mm",
                         "Industrial_Proximity_km", "Elevation_m"]:
                contributing = fval < threshold
            else:
                contributing = fval > threshold

            if prediction == "High" and contributing:
                reasons.append(f"**{readable}** = {fval}{unit} — {explanation}")
            elif prediction == "Low" and not contributing:
                reasons.append(f"**{readable}** = {fval}{unit} — favorable for lower heat levels")
            else:
                reasons.append(f"**{readable}** = {fval}{unit} (importance: {feature_importances[idx]:.3f})")

    return reasons


# ══════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size: 3rem;">🌍</div>
        <h2 style="font-size:1.3rem; background: linear-gradient(90deg, #00d2ff, #e94560);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 8px 0 0;">UHI Monitor</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Home", "🔮  Predict Heat Level", "📊  Analytics", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; opacity:0.3; font-size:0.9rem; padding:10px 0;">
        AI-Driven UHI Framework<br>v1.0 • 2026
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠  Home":

    # Hero
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">AI-Driven Urban Heat Island<br>Monitoring & Mitigation</h1>
        <p class="hero-subtitle">
            Predict urban heat intensity and get mitigation recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    if model_loaded:
        try:
            df_train = pd.read_csv(os.path.join("data", "X_train.csv"))
            n_train = len(df_train)
            n_features = len(feature_names)
        except:
            n_train = 800
            n_features = 15




        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-item">
                <div class="stat-value" style="font-size:3.8rem;">&nbsp;</div>
                <div class="stat-label" style="font-size:1.2rem;">&nbsp;</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="font-size:3.8rem;">&nbsp;</div>
                <div class="stat-label" style="font-size:1.2rem;">&nbsp;</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="font-size:3.8rem;">&nbsp;</div>
                <div class="stat-label" style="font-size:1.2rem;">&nbsp;</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="font-size:3.8rem;">&nbsp;</div>
                <div class="stat-label" style="font-size:1.2rem;">&nbsp;</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="icon">🌡️</div>
            <h3>Heat Level Prediction</h3>
            <p>Enter environmental parameters and get instant predictions — Low, Medium, or High heat intensity — powered by a Random Forest model.</p>
        </div>
        <div class="feature-card">
            <div class="icon">🔍</div>
            <h3>Explainable AI</h3>
            <p>Understand exactly why a prediction was made. See the top contributing factors with feature importance analysis.</p>
        </div>
        <div class="feature-card">
            <div class="icon">🌿</div>
            <h3>Mitigation Strategies</h3>
            <p>Get tailored, actionable strategies — tree planting, cool roofs, green belts, rainwater harvesting, and more.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA
    st.markdown("""
    <div class="cta-container">
        <p style="opacity: 0.5; font-size: 1.2rem;">
            👈 Select <b>"🔮 Predict Heat Level"</b> from the sidebar to start analyzing
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

    # What is UHI section
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #e94560; margin-top:0;">🏙️ What is the Urban Heat Island Effect?</h3>
        <p style="opacity:0.7; line-height:1.8; margin:0;">
            Urban areas experience significantly higher temperatures than surrounding rural areas — a phenomenon
            known as the <b>Urban Heat Island (UHI)</b> effect. Dense buildings absorb and re-radiate solar heat,
            reduced vegetation limits evaporative cooling, vehicle exhaust and industrial activity release
            anthropogenic heat, and dark pavements absorb more solar energy. This framework uses AI to
            quantify these effects and recommend evidence-based mitigation measures.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    col1, col2, col3, col4 = st.columns(4)
    steps = [
        ("1️⃣", "Collect Data", "Environmental parameters like temperature, humidity, wind speed, green cover, and more."),
        ("2️⃣", "Preprocess", "Balance classes using temperature percentiles. Scale and encode features."),
        ("3️⃣", "ML Prediction", "Random Forest classifies heat intensity as Low, Medium, or High."),
        ("4️⃣", "Mitigate", "Get tailored strategies: trees, cool roofs, green belts, rainwater harvesting."),
    ]
    for col, (num, title, desc) in zip([col1, col2, col3, col4], steps):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; min-height:200px;">
                <div style="font-size:2.4rem; margin-bottom:10px;">{num}</div>
                <h4 style="color:#f0f0f0; margin:0 0 10px; font-size:1.2rem;">{title}</h4>
                <p style="font-size:1.02rem; opacity:0.5; margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: PREDICT HEAT LEVEL
# ══════════════════════════════════════════════════════════════
elif page == "🔮  Predict Heat Level":

    if not model_loaded:
        st.error(f"Could not load model artifacts: {model_error}")
        st.info("Run `python generate_dataset.py && python preprocess.py && python train_model.py` first.")
        st.stop()

    # Page header
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <h1 style="font-size:2.8rem; background: linear-gradient(90deg, #00d2ff, #f9a825, #ff1744);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800;">🔮 Predict Heat Level</h1>
        <p style="opacity:0.5; max-width:620px; margin:0 auto; font-size:1.25rem;">
            Enter environmental parameters below to predict the urban heat intensity level
            and receive tailored mitigation strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

    # ── Auto-Fetch Live Location Data ────────────────────────
    st.markdown('<div class="section-header">📍 Auto-Fill From a Real Location</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='opacity:0.55; margin-top:-8px;'>Urban heat islands are about "
        "<i>differences within a city</i> — a crowded downtown vs. a leafy suburb — "
        "not one number for an entire city. For that to show up, type a specific "
        "<b>neighborhood or area</b> rather than just the city name (e.g. "
        "\"Koregaon Park, Pune\" instead of just \"Pune\"). Weather and air quality "
        "are pulled live; building density, greenery, land use and a few other "
        "fields are estimated from real OpenStreetMap data around that exact spot. "
        "Every slider stays editable afterward.</p>",
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("location_query", "Koregaon Park, Pune")

    loc_col1, loc_col2 = st.columns([3, 1])
    with loc_col1:
        location_query = st.text_input(
            "🌍 Location", key="location_query", label_visibility="collapsed",
            placeholder="e.g. Koregaon Park, Pune"
        )
    with loc_col2:
        fetch_clicked = st.button("⚡ Auto-Fill", use_container_width=True)

    if fetch_clicked:
        with st.spinner("Fetching live data for this location..."):
            geo = geocode_place(location_query)
            if geo is None:
                st.error(f"Couldn't find '{location_query}'. Try a more specific name (e.g. add country).")
            else:
                weather = fetch_weather(geo["lat"], geo["lon"])
                osm = fetch_osm_features(geo["lat"], geo["lon"])

                # Live values, clamped to each slider's valid range so an
                # unusual real-world reading (e.g. a cold snap below 10°C)
                # can't push a slider outside its bounds.
                if weather["temperature"] is not None:
                    st.session_state["temperature"] = float(np.clip(weather["temperature"], 10.0, 55.0))
                if weather["humidity"] is not None:
                    st.session_state["humidity"] = float(np.clip(weather["humidity"], 10.0, 100.0))
                if weather["wind_speed"] is not None:
                    st.session_state["wind_speed"] = float(np.clip(weather["wind_speed"], 0.0, 50.0))
                if weather["rainfall"] is not None:
                    st.session_state["rainfall"] = float(np.clip(weather["rainfall"], 0.0, 400.0))
                if weather["solar_radiation"] is not None:
                    st.session_state["sunshine_level"] = closest_label(
                        weather["solar_radiation"], sunshine_map
                    )
                if weather["elevation"] is not None:
                    st.session_state["elevation_level"] = closest_label(
                        weather["elevation"], elevation_map
                    )
                if weather["aqi"] is not None:
                    st.session_state["air_quality"] = closest_label(weather["aqi"], aqi_map)

                # Estimated values (from OpenStreetMap-derived heuristics) —
                # only applied if the Overpass servers actually responded.
                if osm["ok"]:
                    derived = derive_urban_estimates(osm)
                    st.session_state["green_level"] = closest_label(derived["green_cover"], green_map)
                    st.session_state["building_level"] = closest_label(derived["building_density"], building_map)
                    st.session_state["population_level"] = closest_label(derived["population_density"], pop_map)
                    st.session_state["traffic_level"] = closest_label(derived["traffic_density"], traffic_map)
                    st.session_state["factory_level"] = closest_label(derived["industrial_distance_km"], factory_map)
                    st.session_state["surface_color"] = closest_label(derived["albedo"], albedo_map)
                    st.session_state["vegetation_health"] = closest_label(derived["ndvi"], ndvi_map)
                    st.session_state["land_use_display"] = reverse_land_use_options.get(
                        derived["land_use"], "🏘️ Mix of Everything"
                    )

                st.session_state["last_fetch_debug"] = {"weather": weather, "osm": osm}
                st.session_state["last_fetch_osm_ok"] = osm["ok"]
                st.session_state["last_fetch_label"] = geo["label"]
                st.session_state["last_fetch_latlon"] = (geo["lat"], geo["lon"])
                st.rerun()

    if st.session_state.get("last_fetch_label"):
        st.success(
            f"✅ Auto-filled from **{st.session_state['last_fetch_label']}** — "
            "scroll down to review or tweak any value."
        )
        if st.session_state.get("last_fetch_osm_ok") is False:
            warn_col1, warn_col2 = st.columns([4, 1])
            with warn_col1:
                st.warning(
                    "⚠️ Weather data was fetched, but OpenStreetMap's free map-data servers "
                    "were too busy to respond just now (they're a shared community resource, "
                    "not a dedicated service) — so building density, greenery, land use, traffic "
                    "and population were **left unchanged** instead of being overwritten with guesses."
                )
            with warn_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                retry_clicked = st.button("🔁 Retry", use_container_width=True, key="retry_osm")
            if retry_clicked and st.session_state.get("last_fetch_latlon"):
                with st.spinner("Retrying map data..."):
                    lat, lon = st.session_state["last_fetch_latlon"]
                    osm = fetch_osm_features(lat, lon, force_refresh=True)
                    if osm["ok"]:
                        derived = derive_urban_estimates(osm)
                        st.session_state["green_level"] = closest_label(derived["green_cover"], green_map)
                        st.session_state["building_level"] = closest_label(derived["building_density"], building_map)
                        st.session_state["population_level"] = closest_label(derived["population_density"], pop_map)
                        st.session_state["traffic_level"] = closest_label(derived["traffic_density"], traffic_map)
                        st.session_state["factory_level"] = closest_label(derived["industrial_distance_km"], factory_map)
                        st.session_state["surface_color"] = closest_label(derived["albedo"], albedo_map)
                        st.session_state["vegetation_health"] = closest_label(derived["ndvi"], ndvi_map)
                        st.session_state["land_use_display"] = reverse_land_use_options.get(
                            derived["land_use"], "🏘️ Mix of Everything"
                        )
                    st.session_state["last_fetch_osm_ok"] = osm["ok"]
                    st.session_state["last_fetch_debug"]["osm"] = osm
                    st.rerun()
        with st.expander("🔧 Debug: raw data fetched for this location"):
            st.json(st.session_state.get("last_fetch_debug", {}))


    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input Form ───────────────────────────────────────────
    st.markdown('<div class="section-header">🎛️ Tell Us About Your Area</div>', unsafe_allow_html=True)

    # Defaults — only applied the very first time a key doesn't exist yet,
    # so Auto-Fill (or the user's own edits) always take priority afterward.
    _defaults = {
        "temperature": 35.0, "humidity": 55.0, "wind_speed": 12.0, "rainfall": 80.0,
        "sunshine_level": "Sunny", "elevation_level": "Slightly Elevated",
        "green_level": "Some Parks", "building_level": "Moderate", "population_level": "Medium Density",
        "traffic_level": "Moderate Traffic", "factory_level": "Moderate Distance",
        "land_use_display": "🏠 Houses & Apartments",
        "surface_color": "Dark (Grey)", "vegetation_health": "Some Greenery", "air_quality": "Moderate",
    }
    for _k, _v in _defaults.items():
        st.session_state.setdefault(_k, _v)

    # Group 1: Weather
    st.markdown("""<div class="input-section"><h3>🌤️ What's the Weather Like?</h3></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        temperature = st.slider(
            "🌡️ How hot is it outside? (°C)",
            10.0, 55.0, step=0.5, key="temperature",
            help="Current air temperature. 25°C = pleasant, 35°C = hot summer day, 45°C+ = extreme heat"
        )
    with c2:
        humidity = st.slider(
            "💧 How humid / sticky does it feel? (%)",
            10.0, 100.0, step=1.0, key="humidity",
            help="Low (20%) = dry desert, Medium (50%) = comfortable, High (80%+) = very sticky & sweaty"
        )
    with c3:
        wind_speed = st.slider(
            "💨 How windy is it? (km/h)",
            0.0, 50.0, step=0.5, key="wind_speed",
            help="0 = no breeze, 10 = light breeze, 25 = strong wind, 40+ = very windy"
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        sunshine_level = st.select_slider(
            "☀️ How sunny is it today?",
            options=list(sunshine_map.keys()), key="sunshine_level",
            help="How strong the sunlight feels"
        )
        solar_radiation = sunshine_map[sunshine_level]
    with c5:
        rainfall = st.slider(
            "🌧️ Recent rainfall (mm)",
            0.0, 400.0, step=5.0, key="rainfall",
            help="0 = no rain, 50 = light showers, 150 = moderate rain, 300+ = heavy monsoon"
        )
    with c6:
        elevation_level = st.select_slider(
            "⛰️ Is your area flat or hilly?",
            options=list(elevation_map.keys()), key="elevation_level",
            help="How high above sea level your area is"
        )
        elevation = elevation_map[elevation_level]

    # Group 2: Neighbourhood
    st.markdown("""<div class="input-section"><h3>🏙️ What's Your Neighbourhood Like?</h3></div>""", unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        green_level = st.select_slider(
            "🌿 How much greenery do you see around?",
            options=list(green_map.keys()), key="green_level",
            help="Parks, trees, gardens, grass — how green is the area?"
        )
        green_cover = green_map[green_level]
    with c8:
        building_level = st.select_slider(
            "🏢 How crowded are buildings in your area?",
            options=list(building_map.keys()), key="building_level",
            help="Think about how tightly packed the buildings are around you"
        )
        building_density = building_map[building_level]
    with c9:
        population_level = st.select_slider(
            "👥 How many people live in your area?",
            options=list(pop_map.keys()), key="population_level",
            help="Village = Very Few, Small town = Low, City = Medium, Metro = Crowded, Mumbai/Delhi = Very Crowded"
        )
        population_density = pop_map[population_level]

    c10, c11, c12 = st.columns(3)
    with c10:
        traffic_level = st.select_slider(
            "🚗 How bad is the traffic?",
            options=list(traffic_map.keys()), key="traffic_level",
            help="Think about rush hour — is it usually smooth or always jammed?"
        )
        traffic_density = traffic_map[traffic_level]
    with c11:
        factory_level = st.select_slider(
            "🏭 How close are factories / industries?",
            options=list(factory_map.keys()), key="factory_level",
            help="Are there factories, power plants, or industrial areas near you?"
        )
        industrial_proximity = factory_map[factory_level]
    with c12:
        land_use_display = st.selectbox(
            "🏘️ What type of area is it?",
            options=list(land_use_options.keys()), key="land_use_display",
            help="What is your area mostly used for?"
        )
        land_use = land_use_options[land_use_display]

    # Group 3: Environment
    st.markdown("""<div class="input-section"><h3>🌍 A Few More Details About the Environment</h3></div>""", unsafe_allow_html=True)
    c13, c14, c15 = st.columns(3)
    with c13:
        surface_color = st.select_slider(
            "🏗️ What color are the roads & rooftops?",
            options=list(albedo_map.keys()), key="surface_color",
            help="Dark surfaces absorb heat, light surfaces reflect it. Think about your roads and rooftops."
        )
        albedo = albedo_map[surface_color]
    with c14:
        vegetation_health = st.select_slider(
            "🌱 How green & healthy are the plants?",
            options=list(ndvi_map.keys()), key="vegetation_health",
            help="Look at the trees and grass — are they green and thriving, or dry and brown?"
        )
        ndvi = ndvi_map[vegetation_health]
    with c15:
        air_quality = st.select_slider(
            "😷 How's the air quality?",
            options=list(aqi_map.keys()), key="air_quality",
            help="Can you see smog? Is it hard to breathe outside? Are your eyes burning?"
        )
        aqi = aqi_map[air_quality]

    st.markdown("<br>", unsafe_allow_html=True)

    # Predict button
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        predict_button = st.button(
            "🔮  Predict Heat Level & Get Mitigation Plan",
            use_container_width=True,
            type="primary"
        )

    # ── Prediction Logic ─────────────────────────────────────
    if predict_button:
        input_dict = {
            "Temperature_C": temperature,
            "Humidity_%": humidity,
            "Wind_Speed_kmh": wind_speed,
            "Green_Cover_%": green_cover,
            "Building_Density": building_density,
            "Population_Density_per_km2": population_density,
            "Traffic_Density": traffic_density,
            "Industrial_Proximity_km": industrial_proximity,
            "Albedo": albedo,
            "NDVI": ndvi,
            "Air_Quality_Index": aqi,
            "Solar_Radiation_Wm2": solar_radiation,
            "Rainfall_mm": rainfall,
            "Elevation_m": elevation,
        }

        if land_enc is not None and "Land_Use_Type_Encoded" in feature_names:
            input_dict["Land_Use_Type_Encoded"] = land_enc.transform([land_use])[0]

        input_df = pd.DataFrame([{f: input_dict.get(f, 0) for f in feature_names}])
        input_scaled = scaler.transform(input_df)
        # Keep the scaled values in a DataFrame (not a bare array) so both
        # the model and the SHAP explainer see the same feature names the
        # model was originally trained with — avoids a harmless-but-noisy
        # "X does not have valid feature names" warning.
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

        pred_encoded = model.predict(input_scaled_df)[0]
        pred_proba = model.predict_proba(input_scaled_df)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]

        # ── Explainable AI: SHAP-driven mitigation selection ──
        # This replaces "pick strategies from a fixed list keyed only by
        # the predicted class" with "pick strategies based on which
        # SPECIFIC features actually drove THIS prediction," using SHAP
        # (SHapley Additive exPlanations) — see explainable_mitigation.py.
        shap_result = None
        if shap_loaded:
            try:
                shap_result = explain_and_select_mitigation(
                    model=model,
                    explainer=shap_explainer,
                    input_df=input_scaled_df,
                    feature_names=feature_names,
                    le=le,
                    predicted_class_idx=pred_encoded,
                )
            except Exception:
                shap_result = None  # fall back to the static MITIGATION table below

        # ── Results ──────────────────────────────────────────
        st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

        heat_class = pred_label.lower()
        emoji_map = {"low": "❄️", "medium": "🌤️", "high": "🔥"}
        color_map = {"low": "#00d2ff", "medium": "#f9a825", "high": "#ff1744"}

        # Big result card
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Heat Level</div>
            <div class="result-value heat-{heat_class}">
                {emoji_map.get(heat_class, "🌡️")} {pred_label.upper()}
            </div>
            <div class="result-confidence">Confidence: {pred_proba[pred_encoded]*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability gauges
        col1, col2, col3 = st.columns(3)
        for i, (cls_name, prob) in enumerate(zip(le.classes_, pred_proba)):
            cls_color = color_map.get(cls_name.lower(), "#aaa")
            with [col1, col2, col3][i]:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": f"{cls_name}", "font": {"size": 18, "color": "white"}},
                    number={"suffix": "%", "font": {"color": "white", "size": 28}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"color": "rgba(255,255,255,0.5)"}},
                        "bar": {"color": cls_color},
                        "bgcolor": "rgba(255,255,255,0.05)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 33], "color": "rgba(255,255,255,0.02)"},
                            {"range": [33, 67], "color": "rgba(255,255,255,0.04)"},
                            {"range": [67, 100], "color": "rgba(255,255,255,0.06)"},
                        ],
                    }
                ))
                fig.update_layout(
                    height=220,
                    margin=dict(t=60, b=20, l=30, r=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"}
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Prediction Reasons ──────────────────────────────
        st.markdown('<div class="section-header">📋 Why This Prediction?</div>', unsafe_allow_html=True)
        reasons = get_prediction_reasons(
            input_df, pred_label,
            model.feature_importances_, feature_names
        )
        for reason in reasons:
            st.markdown(f"""
            <div class="reason-box">
                <h4>🔍 Contributing Factor</h4>
                {reason}
            </div>
            """, unsafe_allow_html=True)

        # Optional: real, per-prediction SHAP contributing factors, shown
        # alongside the rule-based reasons above for extra rigor.
        if shap_result is not None:
            with st.expander("🧠 See exact SHAP values for this prediction"):
                for feat, val in shap_result["contributing_factors"][:8]:
                    direction = "⬆️ increases" if val > 0 else "⬇️ decreases"
                    st.write(f"- **{feat}**: {direction} likelihood of "
                             f"'{shap_result['predicted_class']}' (SHAP = {val:+.4f})")

        # Feature importance chart
        st.markdown("<br>", unsafe_allow_html=True)
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#00d2ff", "#f9a825", "#ff1744"],
            title="Feature Importance for This Model"
        )
        fig_imp.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            xaxis={"gridcolor": "rgba(255,255,255,0.05)"},
            yaxis={"gridcolor": "rgba(255,255,255,0.05)"},
            coloraxis_showscale=False,
            title_font_size=18,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # ── Mitigation Suggestions ──────────────────────────
        st.markdown('<div class="section-header">🛡️ AI-Explained Mitigation Strategies</div>', unsafe_allow_html=True)

        if shap_result is not None:
            st.caption(
                "🧠 These recommendations are generated using SHAP (SHapley Additive "
                "exPlanations) — a standard explainable-AI technique that identifies "
                "which specific factors are actually driving *this* prediction, rather "
                "than a fixed list based only on the predicted class."
            )
            cols = st.columns(2)
            for i, (icon, title, desc) in enumerate(shap_result["shap_mitigation"]):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="mitigation-box">
                        <h4>{icon} {title}</h4>
                        {desc}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Fallback: SHAP unavailable for some reason — use the static,
            # severity-tier-only mitigation table instead so the section
            # is never empty.
            st.caption("⚠️ SHAP explainer unavailable this run — showing general "
                       "severity-tier strategies instead.")
            strategies = MITIGATION.get(pred_label, MITIGATION["Medium"])
            cols = st.columns(2)
            for i, strategy in enumerate(strategies):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="mitigation-box">
                        <h4>{strategy['icon']} {strategy['title']}</h4>
                        {strategy['desc']}
                    </div>
                    """, unsafe_allow_html=True)

        # ── Input summary ───────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 View Your Input Summary", expanded=False):
            summary_df = pd.DataFrame({
                "Parameter": list(input_dict.keys()),
                "Value": list(input_dict.values())
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "📊  Analytics":

    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <h1 style="font-size:2.8rem; background: linear-gradient(90deg, #00d2ff, #f9a825, #ff1744);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800;">📊 Model Analytics</h1>
        <p style="opacity:0.5; max-width:620px; margin:0 auto; font-size:1.25rem;">
            Explore the model's performance metrics, confusion matrix, and feature importance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

    if not model_loaded:
        st.error("Model not loaded. Please train the model first.")
        st.stop()

    # Classification report
    report_path = os.path.join("outputs", "classification_report.txt")
    if os.path.exists(report_path):
        st.markdown('<div class="section-header">📋 Classification Report</div>', unsafe_allow_html=True)
        with open(report_path, "r") as f:
            report_text = f.read()
        st.code(report_text, language="text")

    # Two columns for images
    col1, col2 = st.columns(2)

    cm_path = os.path.join("outputs", "confusion_matrix.png")
    fi_path = os.path.join("outputs", "feature_importance.png")

    with col1:
        st.markdown('<div class="section-header">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)
        else:
            st.info("Run `python train_model.py` to generate the confusion matrix.")

    with col2:
        st.markdown('<div class="section-header">📊 Feature Importance</div>', unsafe_allow_html=True)
        if os.path.exists(fi_path):
            st.image(fi_path, use_column_width=True)
        else:
            st.info("Run `python train_model.py` to generate feature importance chart.")

    # Model comparison (Random Forest vs. Logistic Regression baseline)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚖️ Model Comparison</div>', unsafe_allow_html=True)
    mc_path = os.path.join("outputs", "model_comparison.png")
    if os.path.exists(mc_path):
        st.image(mc_path, use_column_width=True)
        mc_report_path = os.path.join("outputs", "model_comparison_report.txt")
        if os.path.exists(mc_report_path):
            with open(mc_report_path, "r") as f:
                st.code(f.read(), language="text")
    else:
        st.info("Run `python train_model.py` to generate the model comparison chart.")

    # Interactive feature importance
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔬 Interactive Feature Importance</div>', unsafe_allow_html=True)

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig = px.bar(
        imp_df, x="Feature", y="Importance",
        color="Importance",
        color_continuous_scale=["#00d2ff", "#f9a825", "#ff1744"],
    )
    fig.update_layout(
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "size": 12},
        xaxis={"gridcolor": "rgba(255,255,255,0.05)", "tickangle": -45},
        yaxis={"gridcolor": "rgba(255,255,255,0.05)"},
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dataset stats
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Dataset Statistics</div>', unsafe_allow_html=True)

    try:
        raw_df = pd.read_csv(os.path.join("data", "uhi_dataset.csv"))
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color:#00d2ff;">{len(raw_df)}</h2>
                <p>Total Samples</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color:#f9a825;">{raw_df.shape[1]}</h2>
                <p>Original Features</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color:#2ecc71;">{raw_df['Temperature_C'].mean():.1f}°C</h2>
                <p>Mean Temperature</p>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color:#e94560;">{raw_df['Temperature_C'].max():.1f}°C</h2>
                <p>Max Temperature</p>
            </div>
            """, unsafe_allow_html=True)

        # Temperature distribution
        st.markdown("<br>", unsafe_allow_html=True)
        fig_hist = px.histogram(
            raw_df, x="Temperature_C", nbins=40,
            title="Temperature Distribution",
            color_discrete_sequence=["#e94560"]
        )
        fig_hist.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            xaxis={"gridcolor": "rgba(255,255,255,0.05)"},
            yaxis={"gridcolor": "rgba(255,255,255,0.05)"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    except Exception:
        st.info("Raw dataset not found. Run `python generate_dataset.py` first.")


# ══════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️  About":

    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <h1 style="font-size:2.8rem; background: linear-gradient(90deg, #00d2ff, #f9a825, #ff1744);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800;">ℹ️ About This Project</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #e94560; margin-top:0;">🎯 Project Overview</h3>
        <p style="opacity:0.75; line-height:1.8;">
            <b>AI-Driven Urban Heat Island Monitoring & Mitigation Planning Framework</b> is a complete
            machine learning project that predicts urban heat intensity and recommends evidence-based
            mitigation strategies for sustainable city planning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00d2ff; margin-top:0;">🧠 ML Pipeline</h3>
            <ul style="opacity:0.7; line-height:2;">
                <li><b>Algorithm:</b> Random Forest (200 trees)</li>
                <li><b>Comparison Baseline:</b> Logistic Regression</li>
                <li><b>Explainability:</b> SHAP (SHapley Additive exPlanations)</li>
                <li><b>Class Balancing:</b> Temperature percentile binning</li>
                <li><b>Extra Safeguard:</b> class_weight='balanced'</li>
                <li><b>Features:</b> 15 environmental parameters</li>
                <li><b>Classes:</b> Low, Medium, High</li>
                <li><b>Scaling:</b> StandardScaler</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #2ecc71; margin-top:0;">🛡️ Mitigation Strategies</h3>
            <ul style="opacity:0.7; line-height:2;">
                <li>🌳 Urban afforestation & green belts</li>
                <li>🏠 Cool roofs & green roofs</li>
                <li>💧 Rainwater harvesting systems</li>
                <li>🌿 Urban green corridors</li>
                <li>🛣️ Permeable pavements</li>
                <li>🚗 Traffic & emission controls</li>
            </ul>
            <p style="opacity:0.55; font-size:1.05rem; margin-top:14px;">
                Strategies for Medium/High predictions are now selected per-prediction
                using SHAP, based on which specific features are driving that result —
                not a fixed list based only on the predicted class.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #f9a825; margin-top:0;">📊 Features Used</h3>
        <table style="width:100%; opacity:0.75; border-collapse:collapse;">
            <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                <td style="padding:8px;"><b>Temperature_C</b></td>
                <td style="padding:8px;">Ambient temperature (°C)</td>
                <td style="padding:8px;"><b>Humidity_%</b></td>
                <td style="padding:8px;">Relative humidity (%)</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                <td style="padding:8px;"><b>Wind_Speed_kmh</b></td>
                <td style="padding:8px;">Wind speed (km/h)</td>
                <td style="padding:8px;"><b>Green_Cover_%</b></td>
                <td style="padding:8px;">Vegetation cover (%)</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                <td style="padding:8px;"><b>Building_Density</b></td>
                <td style="padding:8px;">Building density index</td>
                <td style="padding:8px;"><b>Population_Density</b></td>
                <td style="padding:8px;">People per km²</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                <td style="padding:8px;"><b>Traffic_Density</b></td>
                <td style="padding:8px;">Traffic density index</td>
                <td style="padding:8px;"><b>Industrial_Proximity</b></td>
                <td style="padding:8px;">Distance to industry (km)</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                <td style="padding:8px;"><b>Albedo</b></td>
                <td style="padding:8px;">Surface reflectivity</td>
                <td style="padding:8px;"><b>NDVI</b></td>
                <td style="padding:8px;">Vegetation index</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                <td style="padding:8px;"><b>Air_Quality_Index</b></td>
                <td style="padding:8px;">AQI value</td>
                <td style="padding:8px;"><b>Solar_Radiation</b></td>
                <td style="padding:8px;">Solar radiation (W/m²)</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Rainfall_mm</b></td>
                <td style="padding:8px;">Rainfall (mm)</td>
                <td style="padding:8px;"><b>Elevation_m</b></td>
                <td style="padding:8px;">Altitude (m)</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #e94560; margin-top:0;">📜 Data Source</h3>
        <p style="opacity:0.7; line-height:1.8;">
            Dataset: <a href="https://www.kaggle.com/datasets/atharvasoundankar/urban-heat-island-uhi-monitoring-dataset"
            style="color:#00d2ff;">Urban Heat Island (UHI) Monitoring Dataset</a> on Kaggle<br>
            Author: Atharva Soundankar<br>
            License: Apache 2.0
        </p>
    </div>
    """, unsafe_allow_html=True)

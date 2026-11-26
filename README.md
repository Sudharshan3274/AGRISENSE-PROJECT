# AgriSense KEDS

**Knowledge-Based Crop Recommendation and Agricultural Risk Prediction System**

A Streamlit web application for smart agriculture in Tamil Nadu, India.

## Features

### 🌾 Crop Recommendation (Module 1)
- Machine Learning-based crop prediction using soil and climate inputs
- Top-3 crop suggestions with confidence scores
- Knowledge Engine analysis with agronomic rule validation

### 🌦️ Agricultural Risk Prediction (Module 2)
- **ML Risk Model**: Random Forest trained on 24 years of historical data (2001–2024) across 38 Tamil Nadu districts
- **Rule-Based Engine**: Explainable drought, flood, heat stress, and pest/humidity risk assessment grounded in IMD and TNAU standards
- **Farm Advisories**: Actionable mitigation recommendations per hazard
- **Climate Trends**: Interactive historical rainfall and temperature charts

## Tech Stack
- Python 3.13, Streamlit, scikit-learn, pandas, NumPy, GeoPandas

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Sources
- **NASA POWER** (MERRA-2 Reanalysis) — Temperature, humidity, precipitation
- **IMD** — District-wise daily rainfall observations
- **geoBoundaries** — India ADM2 administrative district boundaries

## Author
Sudharshan — Final Year Project

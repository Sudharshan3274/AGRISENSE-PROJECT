"""
train_risk_model.py
--------------------
Trains and validates a supervised Machine Learning Agricultural Risk Prediction Model
using historical agro-meteorological data for Tamil Nadu (2001–2024).

Methodology:
1. Feature Extraction:
   - Multi-scale rainfall: daily rainfall, 7-day rolling, 30-day rolling, 30-day departure %
   - Dry spell metric: consecutive dry days
   - Thermal & humidity indicators: tmean, tmax, tmin, rh, heat index proxy
   - Temporal seasonality: month of year
2. Target Variable (Agricultural Risk Severity):
   - 'Low': Favorable, safe agro-climatic conditions
   - 'Moderate': Moderate moisture deficit, elevated temperature, or localized humidity
   - 'High': Significant drought stress, heavy rainfall warning, or high heat/pest favorability
   - 'Severe': Extreme drought spell, deluge/flood inundation (>115 mm), or dangerous heatwave
3. Temporal Validation Split:
   - Training Set: 2001–2019 (19 years of historical climate regimes)
   - Testing Set:  2020–2024 (5 years out-of-time evaluation)
4. Evaluation:
   - Multiclass Accuracy, Precision, Recall, Macro F1-Score
   - Feature Importance Analysis (Explainable ML)
   - Serialization to risk_model.joblib and risk_model_metadata.json
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_processor import load_processed_weather, compute_agro_features

MODEL_PATH = Path("risk_model.joblib")
META_PATH = Path("risk_model_metadata.json")

def assign_risk_target(df):
    """
    Assigns rigorous, multi-criteria agricultural risk ground-truth levels:
    - Severe: Daily rain >= 115.5 mm OR (Departure <= -60% and Dry Days >= 21) OR Tmax >= 40.0°C
    - High: Daily rain >= 64.5 mm OR (Departure <= -40% and Dry Days >= 14) OR Tmax >= 38.5°C OR (RH >= 85% and Tmean between 24-32°C)
    - Moderate: Daily rain >= 35.0 mm OR Departure <= -20% OR Tmax >= 37.0°C OR (RH >= 75% and Tmean between 24-32°C)
    - Low: Normal favorable range
    """
    conditions = [
        # Severe
        (df["rainfall"] >= 115.5) | 
        ((df["departure_30d"] <= -60.0) & (df["consecutive_dry_days"] >= 21)) | 
        (df["tmax"] >= 40.0),

        # High
        (df["rainfall"] >= 64.5) | 
        ((df["departure_30d"] <= -40.0) & (df["consecutive_dry_days"] >= 14)) | 
        (df["tmax"] >= 38.5) | 
        ((df["rh"] >= 85.0) & (df["tmean"] >= 24.0) & (df["tmean"] <= 32.0)),

        # Moderate
        (df["rainfall"] >= 35.0) | 
        (df["departure_30d"] <= -20.0) | 
        (df["consecutive_dry_days"] >= 10) |
        (df["tmax"] >= 37.0) | 
        ((df["rh"] >= 75.0) & (df["tmean"] >= 24.0) & (df["tmean"] <= 32.0))
    ]
    
    choices = ["Severe", "High", "Moderate"]
    df["risk_level"] = np.select(conditions, choices, default="Low")
    return df

def train_risk_model():
    print("=" * 60)
    print("AGRISENSE: TRAINING AGRICULTURAL RISK PREDICTION MODEL")
    print("=" * 60)

    # 1. Load full dataset
    print("\n1. Loading and processing 2001–2024 Tamil Nadu historical weather dataset...")
    raw_df = load_processed_weather()
    print(f"Loaded {len(raw_df)} daily records across {raw_df['district'].nunique()} districts.")

    # 2. Compute agro-climatic features
    print("\n2. Computing multi-scale rolling features and dry/wet spells...")
    df = compute_agro_features(raw_df)
    df = assign_risk_target(df)

    print("\nTarget Class Distribution:")
    print(df["risk_level"].value_counts(normalize=True).round(3) * 100)

    # 3. Define feature columns
    feature_cols = [
        "rainfall",
        "rainfall_7d",
        "rainfall_30d",
        "departure_30d",
        "consecutive_dry_days",
        "tmean",
        "tmax",
        "tmin",
        "rh",
        "heat_index",
        "month"
    ]

    # 4. Out-of-time Temporal Split (2001-2019 Train, 2020-2024 Test)
    train_mask = df["year"] <= 2019
    test_mask = df["year"] >= 2020

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "risk_level"]

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, "risk_level"]

    print(f"\n3. Temporal Data Split:")
    print(f"   Training Set (2001–2019): {len(X_train)} samples")
    print(f"   Testing Set  (2020–2024): {len(X_test)} samples")

    # 5. Initialize and train Random Forest Classifier
    print("\n4. Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 6. Evaluation on Out-of-Time Test Set
    print("\n5. Model Evaluation (Out-of-Time Test Set 2020–2024):")
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"   Overall Test Accuracy: {test_acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion Matrix
    labels = ["Low", "Moderate", "High", "Severe"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print("Confusion Matrix:")
    print(cm_df)

    # 7. Feature Importance Analysis
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print("\n6. Feature Importances (Explainable ML Drivers):")
    for feat, imp in feat_imp:
        print(f"   {feat:<22}: {imp * 100:.2f}%")

    # 8. Save Model and Metadata
    print(f"\n7. Serializing model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "features": feature_cols,
        "classes": list(model.classes_),
        "accuracy": float(test_acc),
        "feature_importances": {feat: float(imp) for feat, imp in feat_imp},
        "train_years": "2001-2019",
        "test_years": "2020-2024",
        "districts_count": int(raw_df['district'].nunique())
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {META_PATH}")
    print("\nModel training and validation completed successfully!")

if __name__ == "__main__":
    train_risk_model()

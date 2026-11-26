
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import datetime
from pathlib import Path

from input_validation import validate_inputs
from knowledge_engine import analyze_crop_conditions
from risk_engine import assess_agricultural_risk_rules
from data_processor import load_processed_weather, compute_agro_features, get_district_list


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AgriSense KEDS",
    page_icon="🌱",
    layout="wide"
)


# --------------------------------------------------
# LOAD TRAINED MODELS AND DATASETS
# --------------------------------------------------

@st.cache_resource
def load_crop_model():
    return joblib.load("crop_model.joblib")


@st.cache_data
def load_training_data():
    return pd.read_csv("Crop_recommendation.csv")


@st.cache_resource
def load_risk_model():
    if Path("risk_model.joblib").exists():
        return joblib.load("risk_model.joblib")
    return None


@st.cache_data
def load_risk_metadata():
    if Path("risk_model_metadata.json").exists():
        with open("risk_model_metadata.json", "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_district_options():
    return get_district_list()


try:
    model = load_crop_model()
    training_df = load_training_data()
    risk_model = load_risk_model()
    risk_meta = load_risk_metadata()
    district_list = load_district_options()
except Exception as e:
    st.error(f"Unable to load models or datasets: {e}")
    st.stop()


# --------------------------------------------------
# INITIALIZE SESSION HISTORY
# --------------------------------------------------

if "crop_history" not in st.session_state:
    st.session_state.crop_history = []

if "risk_history" not in st.session_state:
    st.session_state.risk_history = []


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🌱 AgriSense KEDS")
st.sidebar.caption("Smart Farming • Better Tomorrow")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Crop Recommendation",
        "Risk Prediction",
        "Knowledge Base",
        "History"
    ]
)


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.title("🌱 AgriSense KEDS")

st.subheader(
    "Knowledge-Based Crop Recommendation "
    "and Agricultural Risk Prediction"
)

st.write(
    "Make data-driven crop decisions using soil "
    "and climate information."
)

st.divider()


# --------------------------------------------------
# MODULE 1: CROP RECOMMENDATION
# --------------------------------------------------

if page in ["Home", "Crop Recommendation"]:

    st.header("🌾 Crop Recommendation")

    st.write(
        "Enter the soil and environmental conditions "
        "of your agricultural field."
    )

    with st.form("crop_form"):

        col1, col2 = st.columns(2)

        with col1:

            nitrogen = st.number_input(
                "Nitrogen (N)",
                min_value=0.0,
                value=90.0
            )

            phosphorus = st.number_input(
                "Phosphorus (P)",
                min_value=0.0,
                value=42.0
            )

            potassium = st.number_input(
                "Potassium (K)",
                min_value=0.0,
                value=43.0
            )

            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-90.0,
                max_value=60.0,
                value=25.0
            )

        with col2:

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=82.0
            )

            ph = st.number_input(
                "Soil pH",
                min_value=0.0,
                max_value=14.0,
                value=6.5
            )

            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                value=200.0
            )

        submitted = st.form_submit_button(
            "🌱 Get Crop Recommendation",
            use_container_width=True
        )

    if submitted:

        # ------------------------------------------
        # STEP 1: COLLECT INPUT VALUES
        # ------------------------------------------

        values = {
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }

        # ------------------------------------------
        # STEP 2: INPUT VALIDATION
        # ------------------------------------------

        valid, errors = validate_inputs(values)

        if not valid:

            st.error("Please correct these inputs:")

            for error in errors:
                st.write(f"- {error}")

            st.stop()

        st.success("Input validation successful.")

        # ------------------------------------------
        # STEP 3: PREPARE MODEL INPUT
        # ------------------------------------------

        feature_cols = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]

        input_df = pd.DataFrame(
            [[values[col] for col in feature_cols]],
            columns=feature_cols
        )

        # ------------------------------------------
        # STEP 4: CHECK TRAINING DATA RANGES
        # ------------------------------------------

        out_of_range = []

        for feature in feature_cols:

            low = training_df[feature].min()
            high = training_df[feature].max()

            if not low <= values[feature] <= high:

                out_of_range.append(
                    f"{feature}: observed training range "
                    f"is {low:.2f} to {high:.2f}"
                )

        if out_of_range:

            st.warning(
                "Some values are outside the model's "
                "observed training data range. "
                "The prediction may be less reliable."
            )

            for message in out_of_range:
                st.write(f"- {message}")

        # ------------------------------------------
        # STEP 5: MACHINE LEARNING PREDICTION
        # ------------------------------------------

        try:

            prediction = model.predict(input_df)[0]

            probabilities = model.predict_proba(input_df)[0]

            confidence = max(probabilities) * 100

        except Exception as e:

            st.error(f"Prediction failed: {e}")
            st.stop()

        # ------------------------------------------
        # STEP 6: DISPLAY PREDICTION
        # ------------------------------------------

        st.divider()
        st.subheader("🌱 Crop Recommendation Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.success(
                f"Predicted crop: {str(prediction).title()}"
            )

        with result_col2:

            st.metric(
                "Model confidence",
                f"{confidence:.1f}%"
            )

        st.caption(
            "The confidence score represents the model's "
            "estimated probability for its prediction. "
            "It does not guarantee crop suitability or yield."
        )

        # ------------------------------------------
        # STEP 7: SHOW TOP THREE CROP PREDICTIONS
        # ------------------------------------------

        st.subheader("📊 Top Three Model Predictions")

        classes = model.classes_

        ranked = sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True
        )[:3]

        for crop, probability in ranked:

            st.write(
                f"**{str(crop).title()}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

        # ------------------------------------------
        # STEP 8: KNOWLEDGE ENGINE ANALYSIS
        # ------------------------------------------

        st.divider()
        st.subheader("📚 Knowledge Engine Analysis")

        try:

            findings = analyze_crop_conditions(
                values,
                prediction
            )

            if findings:

                for finding in findings:

                    status = finding["status"]
                    message = finding["message"]
                    rule = finding["rule"]

                    if status == "Satisfied":

                        st.success(
                            f"✅ {rule}: {message}"
                        )

                    elif status == "Warning":

                        st.warning(
                            f"⚠️ {rule}: {message}"
                        )

                    else:

                        st.info(
                            f"ℹ️ {rule}: {message}"
                        )

            else:

                st.info(
                    "No applicable knowledge rules were found."
                )

        except Exception as e:

            st.error(
                f"Knowledge Engine analysis failed: {e}"
            )

        st.info(
            "The final crop decision should also consider "
            "local soil conditions, crop season, irrigation, "
            "and advice from agricultural experts."
        )

        # Record crop recommendation in session history
        st.session_state.crop_history.append({
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Top Crop": str(prediction).title(),
            "Confidence": f"{confidence:.1f}%",
            "N": values["N"],
            "P": values["P"],
            "K": values["K"],
            "Temp (°C)": values["temperature"],
            "Humidity (%)": values["humidity"],
            "pH": values["ph"],
            "Rainfall (mm)": values["rainfall"]
        })


# --------------------------------------------------
# MODULE 2: AGRICULTURAL RISK PREDICTION
# --------------------------------------------------

elif page == "Risk Prediction":

    st.header("🌦️ Agricultural Risk Prediction for Tamil Nadu")

    st.write(
        "Assess environmental and climatic agricultural risks across Tamil Nadu districts "
        "using long-term historical data (2001–2024), a validated Machine Learning prediction model, "
        "and an explainable agronomic Rule-Based knowledge engine."
    )

    st.divider()

    # 1. District and Scenario Selection
    col_d1, col_d2 = st.columns([1, 2])

    with col_d1:
        default_index = district_list.index("Thanjavur") if "Thanjavur" in district_list else 0
        selected_district = st.selectbox(
            "Select District (Tamil Nadu)",
            district_list,
            index=default_index
        )

    with col_d2:
        scenario_preset = st.selectbox(
            "Quick Scenario / Simulation Preset",
            [
                "Custom / Live Observations",
                "Severe Summer Drought (Ramanathapuram / Delta)",
                "Extreme Monsoon Deluge / Flood Alert (Cuddalore / Chennai)",
                "Dangerous Heatwave Stress (Vellore / Karur)",
                "High Humidity & Paddy Blast / Pest Alert (Cauvery Delta)",
                "Optimal Favorable Growing Conditions (Coimbatore / Western)"
            ]
        )

    # Preset values dictionary for demonstration and interactive testing
    preset_values = {
        "Custom / Live Observations": {
            "rain": 4.5, "rain30": 85.0, "dep30": -10.0, "dry_days": 3,
            "tmax": 33.0, "tmean": 28.5, "tmin": 24.0, "rh": 72.0, "month": 10
        },
        "Severe Summer Drought (Ramanathapuram / Delta)": {
            "rain": 0.0, "rain30": 12.0, "dep30": -75.0, "dry_days": 25,
            "tmax": 39.5, "tmean": 32.0, "tmin": 27.5, "rh": 48.0, "month": 5
        },
        "Extreme Monsoon Deluge / Flood Alert (Cuddalore / Chennai)": {
            "rain": 145.0, "rain30": 380.0, "dep30": 125.0, "dry_days": 0,
            "tmax": 28.0, "tmean": 25.5, "tmin": 23.0, "rh": 94.0, "month": 11
        },
        "Dangerous Heatwave Stress (Vellore / Karur)": {
            "rain": 0.0, "rain30": 25.0, "dep30": -45.0, "dry_days": 18,
            "tmax": 41.5, "tmean": 34.0, "tmin": 28.0, "rh": 42.0, "month": 5
        },
        "High Humidity & Paddy Blast / Pest Alert (Cauvery Delta)": {
            "rain": 15.0, "rain30": 140.0, "dep30": 15.0, "dry_days": 1,
            "tmax": 31.0, "tmean": 27.5, "tmin": 24.5, "rh": 88.0, "month": 10
        },
        "Optimal Favorable Growing Conditions (Coimbatore / Western)": {
            "rain": 8.0, "rain30": 110.0, "dep30": 5.0, "dry_days": 2,
            "tmax": 30.0, "tmean": 25.0, "tmin": 21.0, "rh": 65.0, "month": 8
        }
    }
    defaults = preset_values[scenario_preset]

    st.subheader(f"📍 Meteorological Inputs for {selected_district}")

    with st.form("risk_form"):
        r_col1, r_col2 = st.columns(2)

        with r_col1:
            st.markdown("##### 🌧️ Rainfall & Moisture Metrics")
            daily_rainfall = st.number_input(
                "Daily Rainfall (mm)",
                min_value=0.0,
                max_value=400.0,
                value=float(defaults["rain"]),
                step=0.5,
                help="Today's 24-hour precipitation amount"
            )

            rainfall_30d = st.number_input(
                "30-Day Cumulative Rainfall (mm)",
                min_value=0.0,
                max_value=1500.0,
                value=float(defaults["rain30"]),
                step=5.0,
                help="Total rainfall accumulated over the past 30 days"
            )

            departure_30d = st.number_input(
                "30-Day Rainfall Departure (% from normal)",
                min_value=-100.0,
                max_value=500.0,
                value=float(defaults["dep30"]),
                step=1.0,
                help="Percentage departure from district 30-day climate baseline. Negative indicates deficit."
            )

            consecutive_dry_days = st.number_input(
                "Consecutive Dry Days (Rainfall < 2.5 mm)",
                min_value=0,
                max_value=120,
                value=int(defaults["dry_days"]),
                step=1,
                help="Number of continuous days with rainfall below 2.5 mm"
            )

        with r_col2:
            st.markdown("##### 🌡️ Thermal & Humidity Metrics")
            tmax = st.number_input(
                "Maximum Temperature (°C)",
                min_value=10.0,
                max_value=55.0,
                value=float(defaults["tmax"]),
                step=0.5
            )

            tmean = st.number_input(
                "Mean Temperature (°C)",
                min_value=10.0,
                max_value=50.0,
                value=float(defaults["tmean"]),
                step=0.5
            )

            tmin = st.number_input(
                "Minimum Temperature (°C)",
                min_value=5.0,
                max_value=40.0,
                value=float(defaults["tmin"]),
                step=0.5
            )

            rh = st.number_input(
                "Relative Humidity (%)",
                min_value=10.0,
                max_value=100.0,
                value=float(defaults["rh"]),
                step=1.0
            )

            month = st.slider(
                "Month of Year",
                min_value=1,
                max_value=12,
                value=int(defaults["month"]),
                help="1=Jan, 5=May (Summer peak), 10=Oct (Northeast Monsoon)"
            )

        risk_submitted = st.form_submit_button(
            "🔍 Evaluate Agricultural Risk",
            use_container_width=True
        )

    if risk_submitted:

        # Heat Index calculation
        heat_idx = round(
            tmax + 0.33 * (rh / 100.0 * 6.105 * np.exp((17.27 * tmax) / (237.7 + tmax))) - 4.0,
            1
        )

        # Prepare feature vector for ML model
        feature_dict = {
            "rainfall": daily_rainfall,
            "rainfall_7d": round(daily_rainfall * 2.5, 2),
            "rainfall_30d": rainfall_30d,
            "departure_30d": departure_30d,
            "consecutive_dry_days": consecutive_dry_days,
            "tmean": tmean,
            "tmax": tmax,
            "tmin": tmin,
            "rh": rh,
            "heat_index": heat_idx,
            "month": month
        }
        feature_df = pd.DataFrame([feature_dict])

        # Evaluate Rule-Based Engine
        rule_input = {
            "rainfall": daily_rainfall,
            "rainfall_30d": rainfall_30d,
            "departure_30d": departure_30d,
            "consecutive_dry_days": consecutive_dry_days,
            "tmax": tmax,
            "tmean": tmean,
            "rh": rh
        }
        rule_result = assess_agricultural_risk_rules(rule_input)

        # Evaluate Machine Learning Model
        if risk_model is not None:
            ml_prediction = risk_model.predict(feature_df)[0]
            ml_probabilities = risk_model.predict_proba(feature_df)[0]
            ml_confidence = max(ml_probabilities) * 100
            classes = risk_model.classes_
            class_prob_map = dict(zip(classes, ml_probabilities))
        else:
            ml_prediction = "Model not found"
            ml_confidence = 0.0
            class_prob_map = {}

        # Log into session history
        st.session_state.risk_history.append({
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "District": selected_district,
            "ML Predicted Risk": ml_prediction,
            "ML Confidence": f"{ml_confidence:.1f}%",
            "Rule-Based Risk": rule_result["overall_rule_risk"],
            "Rainfall (mm)": daily_rainfall,
            "Departure 30d (%)": f"{departure_30d:+.1f}%",
            "Dry Days": consecutive_dry_days,
            "Tmax (°C)": tmax,
            "RH (%)": rh
        })

        st.divider()

        # --------------------------------------------------
        # RESULTS PRESENTATION: SIDE-BY-SIDE TABS
        # --------------------------------------------------

        tab_ml, tab_rules, tab_advisory, tab_trends = st.tabs([
            "🤖 Machine Learning Risk Prediction",
            "📋 Explainable Rule-Based Assessment",
            "💡 Actionable Farm Advisories",
            "📈 District Climate Trends (2001–2024)"
        ])

        # TAB 1: MACHINE LEARNING RISK PREDICTION
        with tab_ml:
            st.subheader("🤖 Supervised Machine Learning Risk Model")

            badge_cols = st.columns([1, 1, 1])

            with badge_cols[0]:
                st.metric("Target District", selected_district)

            with badge_cols[1]:
                if ml_prediction == "Severe":
                    st.error(f"Predicted Risk: {ml_prediction.upper()}")
                elif ml_prediction == "High":
                    st.warning(f"Predicted Risk: {ml_prediction.upper()}")
                elif ml_prediction == "Moderate":
                    st.info(f"Predicted Risk: {ml_prediction.upper()}")
                else:
                    st.success(f"Predicted Risk: {ml_prediction.upper()}")

            with badge_cols[2]:
                st.metric("Model Confidence", f"{ml_confidence:.1f}%")

            st.write("---")
            st.markdown("##### 📊 Prediction Probability Distribution Across Severity Tiers")

            prob_cols = st.columns(4)
            tier_order = ["Low", "Moderate", "High", "Severe"]
            for idx, tier in enumerate(tier_order):
                prob_val = class_prob_map.get(tier, 0.0)
                with prob_cols[idx]:
                    st.write(f"**{tier} Risk**")
                    st.write(f"{prob_val * 100:.1f}%")
                    st.progress(float(prob_val))

            st.write("---")
            st.markdown("##### 🧠 Explainable AI: Key Meteorological Drivers (Feature Importance)")

            if risk_meta and "feature_importances" in risk_meta:
                importances = risk_meta["feature_importances"]
                top_feats = list(importances.items())[:6]
                
                f_cols = st.columns(3)
                for i, (feat_name, feat_val) in enumerate(top_feats):
                    with f_cols[i % 3]:
                        st.caption(f"**{feat_name}**")
                        st.write(f"{feat_val * 100:.1f}% influence")
                        st.progress(min(feat_val * 2.5, 1.0))

            st.info(
                "💡 **Academic Distinction**: The ML Risk Prediction model is a Random Forest Classifier "
                "trained on 19 years (2001–2019) and rigorously tested on 5 years (2020–2024) of out-of-time "
                "historical climate observations. It captures non-linear, multi-variate interactions across "
                "consecutive dry spells, cumulative departure, and heat/humidity conditions."
            )

        # TAB 2: EXPLAINABLE RULE-BASED ASSESSMENT
        with tab_rules:
            st.subheader("📋 Explainable Agronomic Knowledge Engine")

            comp_risk = rule_result["overall_rule_risk"]
            if comp_risk == "Severe":
                st.error(f"Overall Composite Agronomic Risk: **{comp_risk.upper()}**")
            elif comp_risk == "High":
                st.warning(f"Overall Composite Agronomic Risk: **{comp_risk.upper()}**")
            elif comp_risk == "Moderate":
                st.info(f"Overall Composite Agronomic Risk: **{comp_risk.upper()}**")
            else:
                st.success(f"Overall Composite Agronomic Risk: **{comp_risk.upper()}**")

            st.write("Independent Evaluation of the 4 Key Agro-Meteorological Hazards:")

            hazard_grid1, hazard_grid2 = st.columns(2)

            with hazard_grid1:
                # Drought Card
                d_info = rule_result["drought"]
                with st.expander(f"☀️ Drought Risk: {d_info['level']} ({d_info['rule_id']})", expanded=True):
                    if d_info["level"] == "Severe":
                        st.error(d_info["title"])
                    elif d_info["level"] == "Moderate":
                        st.warning(d_info["title"])
                    else:
                        st.success(d_info["title"])
                    st.write(f"**Trigger Logic:** {d_info['explanation']}")
                    st.caption(f"Standard: IMD Agricultural Drought & Moisture Stress Index")

                # Heat Stress Card
                h_info = rule_result["heat_stress"]
                with st.expander(f"🌡️ Heat Stress Risk: {h_info['level']} ({h_info['rule_id']})", expanded=True):
                    if h_info["level"] == "Severe":
                        st.error(h_info["title"])
                    elif h_info["level"] == "Moderate":
                        st.warning(h_info["title"])
                    else:
                        st.success(h_info["title"])
                    st.write(f"**Trigger Logic:** {h_info['explanation']}")
                    st.caption(f"Standard: IMD Thermal Stress & TNAU Agromet Thresholds")

            with hazard_grid2:
                # Excess Rainfall Card
                r_info = rule_result["excess_rain"]
                with st.expander(f"🌧️ Excess Rain / Flood Hazard: {r_info['level']} ({r_info['rule_id']})", expanded=True):
                    if r_info["level"] == "Severe":
                        st.error(r_info["title"])
                    elif r_info["level"] == "Moderate":
                        st.warning(r_info["title"])
                    else:
                        st.success(r_info["title"])
                    st.write(f"**Trigger Logic:** {r_info['explanation']}")
                    st.caption(f"Standard: IMD Rainfall Classification (>64.5mm Heavy, >115.5mm Very Heavy)")

                # Pest & Disease Card
                p_info = rule_result["pest_humidity"]
                with st.expander(f"🐛 Humidity & Pest/Disease Hazard: {p_info['level']} ({p_info['rule_id']})", expanded=True):
                    if p_info["level"] == "High":
                        st.error(p_info["title"])
                    elif p_info["level"] == "Moderate":
                        st.warning(p_info["title"])
                    else:
                        st.success(p_info["title"])
                    st.write(f"**Trigger Logic:** {p_info['explanation']}")
                    st.caption(f"Standard: TNAU Microclimate Disease Favorability Model (Blast, Sheath Blight)")

            st.info(
                "💡 **Academic Distinction**: Unlike black-box machine learning, the Rule-Based Engine applies "
                "deterministic agronomic physics directly compliant with published IMD and TNAU standards. "
                "Every alert provides a traceable causal explanation that agricultural extension officers and farmers can verify."
            )

        # TAB 3: ACTIONABLE FARM ADVISORIES
        with tab_advisory:
            st.subheader(f"💡 Actionable Agricultural Advisories for {selected_district}")

            advisories = rule_result["advisories"]
            if advisories:
                for adv in advisories:
                    with st.container():
                        st.markdown(f"#### ⚠️ {adv['hazard']} (Severity: {adv['level']})")
                        st.markdown(f"**Standard Rule Reference:** `{adv['rule_id']}`")
                        st.success(f"🌱 **Recommended Action:** {adv['advisory']}")
                        st.write("---")
            else:
                st.success("✅ **Normal Field Conditions:** No adverse environmental risks detected. Safe for scheduled irrigation, fertilization, and cultivation operations.")

        # TAB 4: DISTRICT HISTORICAL CLIMATE TRENDS
        with tab_trends:
            st.subheader(f"📈 Historical Climate Profile: {selected_district} (2001–2024)")

            try:
                hist_df = load_processed_weather(district=selected_district)
                
                # Year selector for historical drilldown
                year_choice = st.slider(
                    "Select Historical Year to Inspect",
                    min_value=2001,
                    max_value=2024,
                    value=2024
                )
                
                year_df = hist_df[hist_df["year"] == year_choice].sort_values("date")
                
                t_col1, t_col2, t_col3 = st.columns(3)
                total_rain = year_df["rainfall"].sum()
                avg_tmax = year_df["tmax"].mean()
                avg_rh = year_df["rh"].mean()

                with t_col1:
                    st.metric(f"Total Rainfall ({year_choice})", f"{total_rain:.1f} mm")
                with t_col2:
                    st.metric(f"Mean Max Temp ({year_choice})", f"{avg_tmax:.1f} °C")
                with t_col3:
                    st.metric(f"Mean Humidity ({year_choice})", f"{avg_rh:.1f} %")

                st.markdown(f"##### Daily Rainfall Trend ({year_choice})")
                st.line_chart(year_df.set_index("date")["rainfall"], use_container_width=True)

                st.markdown(f"##### Temperature Envelope ({year_choice})")
                st.line_chart(year_df.set_index("date")[["tmax", "tmin"]], use_container_width=True)

            except Exception as ex:
                st.warning(f"Could not load historical trends for {selected_district}: {ex}")


# --------------------------------------------------
# KNOWLEDGE BASE PAGE
# --------------------------------------------------

elif page == "Knowledge Base":

    st.header("📚 AgriSense Agricultural Knowledge Base")

    st.write(
        "The AgriSense Knowledge Engine applies explicit, auditable rules based on agricultural science, "
        "meteorological standards (IMD), and university extension advisories (TNAU)."
    )

    st.divider()

    tab_k1, tab_k2 = st.tabs([
        "🌾 Crop Suitability Knowledge Rules",
        "🌦️ Agricultural Risk Assessment Rules"
    ])

    with tab_k1:
        st.subheader("Crop Suitability Knowledge Rules")
        st.markdown(
            """
            | Rule ID | Target Crop | Parameter | Threshold Criteria | Agronomic Rationale |
            | :--- | :--- | :--- | :--- | :--- |
            | **RICE-HUMIDITY-01** | Rice / Paddy | Relative Humidity | `RH >= 70%` | Rice requires high ambient humidity during tillering and panicle initiation to prevent spikelet desiccation. |
            | **RICE-RAINFALL-01** | Rice / Paddy | Rainfall | `Rainfall >= 100 mm` | Semi-aquatic crop with heavy water requirement throughout vegetative and reproductive phases. |
            | **TEMP-HEAT-01** | All Crops | Temperature | `Temperature > 40°C` | High thermal stress causes pollen sterility, flower drop, and leaf chlorosis. |
            | **SOIL-PH-01** | All Crops | Soil pH | `0 <= pH <= 14` | Valid agronomic pH scale check; optimal nutrient uptake occurs within pH 6.0–7.5. |
            """
        )

    with tab_k2:
        st.subheader("Agricultural Risk Assessment Rules (IMD & TNAU Grounded)")
        st.markdown(
            r"""
            #### 1. Drought Risk Rules (IMD Criteria)
            - **RULE-DROUGHT-01 (Severe Drought)**: 30-day rainfall departure ≤ −60% **OR** Consecutive dry days ≥ 21.
              - *Mitigation*: Emergency deficit irrigation, 5 cm straw mulching, spray 1% KCl or 3% kaolin anti-transpirant.
            - **RULE-DROUGHT-02 (Moderate Drought)**: 30-day rainfall departure between −25% and −59% **OR** Consecutive dry days ≥ 14.
              - *Mitigation*: Alternate furrow irrigation, weed management to reduce transpiration.
            - **RULE-DROUGHT-03 (Low Drought Risk)**: Departure > −25% and dry spell < 14 days.

            #### 2. Excess Rainfall / Flood Hazard Rules (IMD Criteria)
            - **RULE-RAIN-01 (Severe Flood Inundation)**: Daily rainfall ≥ 115.5 mm (IMD Very Heavy) **OR** (30-day rainfall ≥ 250 mm and departure ≥ +60%).
              - *Mitigation*: Open drainage outlets, withhold synthetic fertilizers, apply 0.5% ZnSO4 post-drainage.
            - **RULE-RAIN-02 (Moderate Waterlogging)**: Daily rainfall between 64.5 mm and 115.4 mm (IMD Heavy Rain).
              - *Mitigation*: Inspect drainage bunds, clear collector ditches.
            - **RULE-RAIN-03 (Normal Rain)**: Daily rainfall < 64.5 mm.

            #### 3. Heat Stress Hazard Rules
            - **RULE-HEAT-01 (Severe Heatwave)**: Maximum Temperature Tmax ≥ 40.0°C.
              - *Mitigation*: Early morning or evening irrigation, micro-sprinklers in orchards.
            - **RULE-HEAT-02 (Moderate Thermal Stress)**: 37.0°C ≤ Tmax < 40.0°C.
              - *Mitigation*: Soil mulching to suppress soil heat penetration.

            #### 4. Humidity & Pest / Disease Hazard Rules
            - **RULE-PEST-01 (High Disease Favorability)**: Relative Humidity RH ≥ 80% **AND** Mean Temperature 24°C ≤ T ≤ 32°C.
              - *Mitigation*: Prophylactic spray of *Pseudomonas fluorescens* (10g/L), scout lower canopy for blast/sheath blight.
            - **RULE-PEST-02 (Moderate Favorability)**: RH ≥ 70%.
            """
        )


# --------------------------------------------------
# MODULE 3: PREDICTION HISTORY
# --------------------------------------------------

elif page == "History":

    st.header("📋 AgriSense Session Query History")

    st.write(
        "Review predictions and evaluations performed during your current active session."
    )

    h_tab1, h_tab2 = st.tabs([
        "🌾 Crop Recommendation History",
        "🌦️ Agricultural Risk History"
    ])

    with h_tab1:
        if st.session_state.crop_history:
            df_crop_hist = pd.DataFrame(st.session_state.crop_history)
            st.dataframe(df_crop_hist, use_container_width=True)
            if st.button("🗑️ Clear Crop History", key="clear_crop"):
                st.session_state.crop_history = []
                st.rerun()
        else:
            st.info("No crop recommendations recorded yet. Run a prediction from the Crop Recommendation page!")

    with h_tab2:
        if st.session_state.risk_history:
            df_risk_hist = pd.DataFrame(st.session_state.risk_history)
            st.dataframe(df_risk_hist, use_container_width=True)
            if st.button("🗑️ Clear Risk History", key="clear_risk"):
                st.session_state.risk_history = []
                st.rerun()
        else:
            st.info("No risk predictions recorded yet. Run an assessment from the Risk Prediction page!")

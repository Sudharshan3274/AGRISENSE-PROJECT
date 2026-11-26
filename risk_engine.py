"""
risk_engine.py
---------------
Explainable Rule-Based Agricultural Risk Assessment Engine.

Encodes deterministic agro-meteorological rules complying with IMD
(India Meteorological Department) and TNAU (Tamil Nadu Agricultural University)
advisory thresholds for:
1. Drought Risk
2. Excess Rainfall / Flood Risk
3. Heat Stress Risk
4. Humidity & Pest / Disease Risk

Distinct from Machine Learning:
- Transparent and fully auditable
- Grounded in established agronomic physics and operational thresholds
- Explains the exact causal triggers and provides actionable farm advisories
"""

def evaluate_drought_risk(departure_30d, consecutive_dry_days):
    if departure_30d <= -60.0 or consecutive_dry_days >= 21:
        return {
            "level": "Severe",
            "rule_id": "RULE-DROUGHT-01",
            "title": "Severe Agricultural Drought",
            "status": "Critical",
            "explanation": (
                f"Rainfall deficit is {departure_30d:.1f}% below 30-day normal with "
                f"{consecutive_dry_days} consecutive dry days. Soil moisture depletion is severe."
            ),
            "advisory": (
                "Trigger contingency irrigation immediately. Apply 5–7 cm straw mulch to arrest "
                "evaporative loss. Spray 1% potassium chloride or kaolin (3%) anti-transpirant. "
                "Prioritize flowering/pod-formation stages and avoid nitrogen top-dressing."
            )
        }
    elif departure_30d <= -25.0 or consecutive_dry_days >= 14:
        return {
            "level": "Moderate",
            "rule_id": "RULE-DROUGHT-02",
            "title": "Moderate Moisture Stress",
            "status": "Warning",
            "explanation": (
                f"Rainfall deficit is {departure_30d:.1f}% with {consecutive_dry_days} dry days. "
                "Crop enters mild-to-moderate vegetative moisture stress."
            ),
            "advisory": (
                "Adopt alternate furrow irrigation or drip deficit scheduling. "
                "Weed the field thoroughly to eliminate root water competition."
            )
        }
    else:
        return {
            "level": "Low",
            "rule_id": "RULE-DROUGHT-03",
            "title": "Adequate Moisture",
            "status": "Normal",
            "explanation": (
                f"Rainfall departure is {departure_30d:+.1f}% and dry spell is {consecutive_dry_days} days. "
                "Root-zone soil moisture is within normal agricultural limits."
            ),
            "advisory": "Maintain regular scheduled irrigation as per crop growth stage."
        }

def evaluate_excess_rain_risk(daily_rain, departure_30d, rainfall_30d):
    if daily_rain >= 115.5 or (rainfall_30d >= 250.0 and departure_30d >= 60.0):
        return {
            "level": "Severe",
            "rule_id": "RULE-RAIN-01",
            "title": "Severe Inundation / Flood Hazard",
            "status": "Critical",
            "explanation": (
                f"Recorded daily precipitation of {daily_rain:.1f} mm exceeds IMD 'Very Heavy Rain' "
                f"threshold (115.5 mm) with 30-day surplus of {departure_30d:+.1f}%. Risk of root asphyxiation and lodging."
            ),
            "advisory": (
                "Open immediate field drainage ditches and clear collector channels. "
                "Withhold synthetic fertilizers and pesticide sprays until fields drain to prevent runoff. "
                "Post-drainage, spray 0.5% zinc sulphate or 1% urea to revive yellowing submerged crops."
            )
        }
    elif daily_rain >= 64.5 or (departure_30d >= 30.0 and daily_rain >= 25.0):
        return {
            "level": "Moderate",
            "rule_id": "RULE-RAIN-02",
            "title": "Heavy Rainfall / Waterlogging Alert",
            "status": "Warning",
            "explanation": (
                f"Daily rainfall of {daily_rain:.1f} mm meets IMD 'Heavy Rain' threshold (64.5 mm). "
                "Surface runoff and soil saturation likely."
            ),
            "advisory": (
                "Ensure bund drainage outlets are open. Check that low-lying field patches do not puddle. "
                "Postpone chemical spraying for at least 24 hours."
            )
        }
    else:
        return {
            "level": "Low",
            "rule_id": "RULE-RAIN-03",
            "title": "Normal Precipitation",
            "status": "Normal",
            "explanation": (
                f"Daily rainfall of {daily_rain:.1f} mm is well within standard soil infiltration capacity."
            ),
            "advisory": "Standard field conditions. Safe for scheduled agronomic operations."
        }

def evaluate_heat_risk(tmax):
    if tmax >= 40.0:
        return {
            "level": "Severe",
            "rule_id": "RULE-HEAT-01",
            "title": "Severe Heatwave Stress",
            "status": "Critical",
            "explanation": (
                f"Maximum temperature reached {tmax:.1f}°C (>= 40°C threshold). "
                "Can cause pollen sterility, flower drop in pulses/cotton, and leaf scorching."
            ),
            "advisory": (
                "Provide light and frequent irrigation during evening or early morning hours. "
                "Operate micro-sprinklers in horticulture crops to lower canopy temperature. "
                "Avoid daytime chemical sprays to prevent phytotoxicity."
            )
        }
    elif tmax >= 37.0:
        return {
            "level": "Moderate",
            "rule_id": "RULE-HEAT-02",
            "title": "Elevated Thermal Stress",
            "status": "Warning",
            "explanation": (
                f"Maximum temperature of {tmax:.1f}°C exceeds optimal physiological threshold (37°C). "
                "Accelerates evapotranspiration rates."
            ),
            "advisory": (
                "Maintain soil cover with biomass mulch to suppress root zone heating. "
                "Shorten irrigation intervals by 2–3 days."
            )
        }
    else:
        return {
            "level": "Low",
            "rule_id": "RULE-HEAT-03",
            "title": "Favorable Thermal Range",
            "status": "Normal",
            "explanation": (
                f"Maximum temperature of {tmax:.1f}°C is within optimal photosynthetic range."
            ),
            "advisory": "Temperature is supportive for vegetative and reproductive growth."
        }

def evaluate_pest_humidity_risk(rh, tmean):
    if rh >= 80.0 and (24.0 <= tmean <= 32.0):
        return {
            "level": "High",
            "rule_id": "RULE-PEST-01",
            "title": "High Microclimate Pest / Disease Hazard",
            "status": "Critical",
            "explanation": (
                f"High relative humidity ({rh:.1f}%) paired with warm temperature ({tmean:.1f}°C) "
                "creates ideal incubation conditions for fungal blast, sheath blight, and sucking pests."
            ),
            "advisory": (
                "Scout fields closely, particularly lower canopy and leaf sheaths. "
                "Apply prophylactic bio-control (Pseudomonas fluorescens @ 10g/L or Neem oil @ 3ml/L). "
                "Refrain from heavy nitrogen application which creates dense, disease-susceptible foliage."
            )
        }
    elif rh >= 70.0:
        return {
            "level": "Moderate",
            "rule_id": "RULE-PEST-02",
            "title": "Moderate Humidity Risk",
            "status": "Warning",
            "explanation": (
                f"Relative humidity is {rh:.1f}%. Potential for localized pest buildup if sustained."
            ),
            "advisory": (
                "Inspect field border rows twice weekly. Keep drainage channels clean to reduce canopy humidity."
            )
        }
    else:
        return {
            "level": "Low",
            "rule_id": "RULE-PEST-03",
            "title": "Low Disease Favorability",
            "status": "Normal",
            "explanation": (
                f"Relative humidity of {rh:.1f}% does not support rapid fungal spore germination."
            ),
            "advisory": "Maintain routine crop monitoring and clean field sanitation."
        }

def assess_agricultural_risk_rules(values):
    """
    Main rule evaluation orchestrator.
    Computes component risk scores and overall composite rating.
    """
    daily_rain = float(values.get("rainfall", 0.0))
    rainfall_30d = float(values.get("rainfall_30d", daily_rain * 30))
    departure_30d = float(values.get("departure_30d", 0.0))
    consecutive_dry_days = int(values.get("consecutive_dry_days", 0))
    tmax = float(values.get("tmax", 30.0))
    tmean = float(values.get("tmean", 27.0))
    rh = float(values.get("rh", 70.0))

    drought = evaluate_drought_risk(departure_30d, consecutive_dry_days)
    excess_rain = evaluate_excess_rain_risk(daily_rain, departure_30d, rainfall_30d)
    heat = evaluate_heat_risk(tmax)
    pest = evaluate_pest_humidity_risk(rh, tmean)

    # Determine overall composite rule risk
    levels = [drought["level"], excess_rain["level"], heat["level"], pest["level"]]
    if "Severe" in levels:
        overall = "Severe"
    elif "High" in levels or levels.count("Moderate") >= 2:
        overall = "High"
    elif "Moderate" in levels:
        overall = "Moderate"
    else:
        overall = "Low"

    advisories = []
    for hazard in [drought, excess_rain, heat, pest]:
        if hazard["level"] in ["Severe", "High", "Moderate"]:
            advisories.append({
                "hazard": hazard["title"],
                "level": hazard["level"],
                "rule_id": hazard["rule_id"],
                "advisory": hazard["advisory"]
            })

    return {
        "overall_rule_risk": overall,
        "drought": drought,
        "excess_rain": excess_rain,
        "heat_stress": heat,
        "pest_humidity": pest,
        "advisories": advisories
    }

if __name__ == "__main__":
    test_input = {
        "rainfall": 125.0,
        "rainfall_30d": 310.0,
        "departure_30d": 75.0,
        "consecutive_dry_days": 0,
        "tmax": 33.0,
        "tmean": 28.5,
        "rh": 92.0
    }
    result = assess_agricultural_risk_rules(test_input)
    print("Rule-Based Assessment Result:")
    print("Overall Risk:", result["overall_rule_risk"])
    print("Flood:", result["excess_rain"]["level"], "-", result["excess_rain"]["explanation"])
    print("Pest:", result["pest_humidity"]["level"], "-", result["pest_humidity"]["explanation"])

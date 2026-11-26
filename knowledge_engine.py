
def analyze_crop_conditions(values, crop):

    findings = []

    humidity = float(values["humidity"])
    rainfall = float(values["rainfall"])
    temperature = float(values["temperature"])
    ph = float(values["ph"])

    # Rule 1: Rice humidity condition
    if crop.lower() == "rice":

        if humidity >= 70:
            findings.append({
                "rule": "RICE-HUMIDITY-01",
                "status": "Satisfied",
                "message": "Humidity meets the demonstration threshold for rice."
            })
        else:
            findings.append({
                "rule": "RICE-HUMIDITY-01",
                "status": "Warning",
                "message": "Humidity is below the demonstration threshold for rice."
            })

        # Rule 2: Rice rainfall condition
        if rainfall >= 100:
            findings.append({
                "rule": "RICE-RAINFALL-01",
                "status": "Satisfied",
                "message": "Rainfall meets the demonstration threshold for rice."
            })
        else:
            findings.append({
                "rule": "RICE-RAINFALL-01",
                "status": "Warning",
                "message": "Rainfall is below the demonstration threshold for rice."
            })

    # Rule 3: High temperature condition
    if temperature > 40:
        findings.append({
            "rule": "TEMP-HEAT-01",
            "status": "Warning",
            "message": "High temperature may create heat stress."
        })

    # Rule 4: Soil pH validity
    if 0 <= ph <= 14:
        findings.append({
            "rule": "SOIL-PH-01",
            "status": "Satisfied",
            "message": "Soil pH is within the valid measurement scale."
        })

    if not findings:
        findings.append({
            "rule": "GENERAL-01",
            "status": "Information",
            "message": "No crop-specific knowledge rule is available yet."
        })

    return findings

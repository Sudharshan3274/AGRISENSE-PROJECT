
def validate_inputs(values):

    errors = []

    required = [
        "N", "P", "K",
        "temperature", "humidity",
        "ph", "rainfall"
    ]

    # Check whether all inputs are available
    for name in required:
        if name not in values:
            errors.append(f"Missing input: {name}")
            continue

        try:
            value = float(values[name])
        except (ValueError, TypeError):
            errors.append(f"{name} must be numeric.")
            continue

        # Check invalid numeric values
        if not (float("-inf") < value < float("inf")):
            errors.append(f"{name} must be a valid number.")
            continue

        # Validate individual parameters
        if name in ["N", "P", "K"] and value < 0:
            errors.append(f"{name} cannot be negative.")

        elif name == "temperature" and not -90 <= value <= 60:
            errors.append("Temperature must be between -90 and 60°C.")

        elif name == "humidity" and not 0 <= value <= 100:
            errors.append("Humidity must be between 0 and 100%.")

        elif name == "ph" and not 0 <= value <= 14:
            errors.append("Soil pH must be between 0 and 14.")

        elif name == "rainfall" and value < 0:
            errors.append("Rainfall cannot be negative.")

    return len(errors) == 0, errors

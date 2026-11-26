"""
extract_tamil_nadu_districts.py
--------------------------------
Extracts the 38 administrative district boundaries of Tamil Nadu
from the India ADM2 geoBoundaries GeoJSON file and calculates
district centroid coordinates (latitude and longitude) for NASA POWER
and IMD spatial weather integration.

Input:  data/india_districts.geojson
Output: data/tamil_nadu_districts.geojson
"""

import os
from pathlib import Path
import geopandas as gpd

# Define paths
DATA_DIR = Path("data")
INPUT_GEOJSON = DATA_DIR / "india_districts.geojson"
OUTPUT_GEOJSON = DATA_DIR / "tamil_nadu_districts.geojson"

# Canonical list of 38 Tamil Nadu districts
TN_DISTRICT_NAMES = [
    "Ariyalur", "Chengalputtu", "Chennai", "Coimbatore", "Cuddalore",
    "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kancheepuram",
    "Kanniyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai",
    "Nagapattinam", "Namakkal", "The Nilgiris", "Perambalur", "Pudukkottai",
    "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi",
    "Thanjavur", "Theni", "Thiruvallur", "Thiruvarur", "Thoothukkudi",
    "Tiruchirappalli", "Tirunelveli", "Tirupathur", "Tiruppur",
    "Tiruvannamalai", "Vellore", "Viluppuram", "Virudhunagar"
]

def extract_tamil_nadu():
    print(f"Loading input GeoJSON: {INPUT_GEOJSON}...")
    if not INPUT_GEOJSON.exists():
        raise FileNotFoundError(f"Input file not found at {INPUT_GEOJSON}")

    gdf = gpd.read_file(INPUT_GEOJSON)
    print(f"Loaded India dataset with {len(gdf)} administrative units.")

    # Spatial bounding box around Tamil Nadu to avoid collisions with homonymous districts
    # Tamil Nadu roughly lies between Lat 8.0°N - 13.6°N and Lon 76.2°E - 80.4°E
    centres = gdf.geometry.representative_point()
    in_box = (
        (centres.x >= 76.0) & (centres.x <= 80.5) &
        (centres.y >= 8.0) & (centres.y <= 13.7)
    )

    # Filter to Tamil Nadu districts
    tn_gdf = gdf[gdf["shapeName"].isin(TN_DISTRICT_NAMES) & in_box].copy()

    # Validate district count
    found_count = len(tn_gdf)
    print(f"Districts matched: {found_count} / {len(TN_DISTRICT_NAMES)}")

    missing = set(TN_DISTRICT_NAMES) - set(tn_gdf["shapeName"])
    if missing:
        print(f"Warning: Missing districts: {missing}")

    # Ensure clean valid geometries
    invalid_mask = ~tn_gdf.geometry.is_valid
    if invalid_mask.sum() > 0:
        print(f"Fixing {invalid_mask.sum()} invalid geometries using buffer(0)...")
        tn_gdf.geometry = tn_gdf.geometry.buffer(0)

    # Compute district representative centroids (lat/lon)
    rep_points = tn_gdf.geometry.representative_point()
    tn_gdf["lat"] = rep_points.y.round(4)
    tn_gdf["lon"] = rep_points.x.round(4)

    # Select standard attributes
    columns_to_keep = ["shapeName", "shapeID", "lat", "lon", "geometry"]
    available_cols = [col for col in columns_to_keep if col in tn_gdf.columns]
    tn_clean = tn_gdf[available_cols].sort_values("shapeName").reset_index(drop=True)

    # Export clean GeoJSON
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tn_clean.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"Successfully saved {len(tn_clean)} districts to {OUTPUT_GEOJSON}")
    print("\nSample extracted districts with coordinates:")
    print(tn_clean[["shapeName", "lat", "lon"]].head(5).to_string(index=False))

if __name__ == "__main__":
    extract_tamil_nadu()

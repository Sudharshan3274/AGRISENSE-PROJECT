"""
fetch_nasa_power.py
--------------------
Utility script to fetch historical daily agroclimatology data
from the official NASA POWER (Prediction Of Worldwide Energy Resources) API
for Tamil Nadu districts (2001–2024).

Parameters fetched:
- T2M: Temperature at 2 Meters (°C)
- T2M_MAX: Maximum Temperature at 2 Meters (°C)
- T2M_MIN: Minimum Temperature at 2 Meters (°C)
- RH2M: Relative Humidity at 2 Meters (%)
- PRECTOTCORR: Corrected Precipitation (mm/day)

Usage:
  python fetch_nasa_power.py --district Chennai --start 20010101 --end 20241231
  python fetch_nasa_power.py --all --start 20010101 --end 20241231
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd
import requests

API_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
GEOJSON_PATH = Path("data") / "tamil_nadu_districts.geojson"
CACHE_DIR = Path("data") / "nasa_power_cache"

def load_districts():
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(f"Missing {GEOJSON_PATH}. Run extract_tamil_nadu_districts.py first.")
    gdf = gpd.read_file(GEOJSON_PATH)
    if "lat" not in gdf.columns or "lon" not in gdf.columns:
        rep = gdf.geometry.representative_point()
        gdf["lat"] = rep.y.round(4)
        gdf["lon"] = rep.x.round(4)
    return gdf

def fetch_district_weather(district_name, lat, lon, start_date="20010101", end_date="20241231"):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{district_name.lower().replace(' ', '_')}_{start_date}_{end_date}.csv"
    
    if cache_file.exists():
        print(f"[{district_name}] Found cached file: {cache_file}")
        return pd.read_csv(cache_file)

    print(f"[{district_name}] Fetching NASA POWER data for ({lat}, {lon}) from {start_date} to {end_date}...")
    
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    try:
        response = requests.get(API_ENDPOINT, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        properties = data["properties"]["parameter"]
        dates = list(properties["T2M"].keys())
        
        records = []
        for d in dates:
            t2m = properties["T2M"].get(d)
            tmax = properties["T2M_MAX"].get(d)
            tmin = properties["T2M_MIN"].get(d)
            rh = properties["RH2M"].get(d)
            prec = properties["PRECTOTCORR"].get(d)

            # NASA missing value is -999
            if any(val == -999 for val in [t2m, tmax, tmin, rh, prec]):
                continue

            records.append({
                "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}",
                "district": district_name,
                "lat": lat,
                "lon": lon,
                "tmean": t2m,
                "tmax": tmax,
                "tmin": tmin,
                "rh": rh,
                "rainfall": prec
            })
            
        df = pd.DataFrame(records)
        df.to_csv(cache_file, index=False)
        print(f"[{district_name}] Downloaded and cached {len(df)} daily records to {cache_file}")
        return df

    except Exception as e:
        print(f"[{district_name}] Failed to fetch from NASA POWER: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Fetch NASA POWER weather data for Tamil Nadu districts.")
    parser.add_argument("--district", type=str, help="Specific district name (e.g., Chennai)")
    parser.add_argument("--all", action="store_true", help="Fetch for all 38 districts")
    parser.add_argument("--start", type=str, default="20010101", help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, default="20241231", help="End date YYYYMMDD")
    args = parser.parse_args()

    gdf = load_districts()

    if args.district:
        matches = gdf[gdf["shapeName"].str.lower() == args.district.lower()]
        if matches.empty:
            print(f"District '{args.district}' not found. Available: {sorted(gdf['shapeName'].tolist())}")
            sys.exit(1)
        row = matches.iloc[0]
        fetch_district_weather(row["shapeName"], row["lat"], row["lon"], args.start, args.end)
    elif args.all:
        all_dfs = []
        for _, row in gdf.iterrows():
            df = fetch_district_weather(row["shapeName"], row["lat"], row["lon"], args.start, args.end)
            if df is not None:
                all_dfs.append(df)
            time.sleep(1) # Be gentle with NASA POWER API
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            out_file = Path("data") / f"tamil_nadu_nasa_power_{args.start}_{args.end}.csv"
            combined.to_csv(out_file, index=False)
            print(f"All districts saved to {out_file} ({len(combined)} rows)")
    else:
        print("Please provide --district <name> or --all. Example: python fetch_nasa_power.py --district Chennai")

if __name__ == "__main__":
    main()

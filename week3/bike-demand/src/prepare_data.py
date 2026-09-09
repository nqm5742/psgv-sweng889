"""Prepare the raw Seoul Bike Sharing dataset for modeling.

Reads the raw CSV (unmodified), parses the date, encodes categorical
variables for scikit-learn, adds a couple of simple date-derived
features, and writes the result to data/processed/bike_demand_prepared.csv.

No row filtering, feature selection, or model training happens here.
"""

import pandas as pd

RAW_PATH = "data/raw/SeoulBikeData.csv"
PROCESSED_PATH = "data/processed/bike_demand_prepared.csv"

RENAME_MAP = {
    "Date": "date",
    "Rented Bike Count": "rented_bike_count",
    "Hour": "hour",
    "Temperature(°C)": "temperature_c",
    "Humidity(%)": "humidity_pct",
    "Wind speed (m/s)": "wind_speed_ms",
    "Visibility (10m)": "visibility_10m",
    "Dew point temperature(°C)": "dew_point_c",
    "Solar Radiation (MJ/m2)": "solar_radiation_mjm2",
    "Rainfall(mm)": "rainfall_mm",
    "Snowfall (cm)": "snowfall_cm",
    "Seasons": "season",
    "Holiday": "holiday",
    "Functioning Day": "functioning_day",
}


def main():
    df = pd.read_csv(RAW_PATH, encoding="cp1252")
    df = df.rename(columns=RENAME_MAP)

    # Parse the date (day/month/year in the raw file) into a real date.
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

    # Simple, clearly-justified date-derived features:
    # - month: finer seasonal granularity than the 4-category `season`.
    # - day_of_week: captures weekday/weekend commuting patterns (Monday=0).
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    # Keep the parsed date as an ISO string for reference / chronological
    # train-test splitting later. It is not itself a numeric model input.
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Binary categoricals -> a single 0/1 column each.
    df["holiday"] = df["holiday"].map({"No Holiday": 0, "Holiday": 1})
    df["functioning_day"] = df["functioning_day"].map({"No": 0, "Yes": 1})

    # Nominal categorical (season, 4 unordered categories) -> one-hot columns.
    df = pd.get_dummies(df, columns=["season"], prefix="season")
    season_cols = [c for c in df.columns if c.startswith("season_")]
    df[season_cols] = df[season_cols].astype(int)

    # Move the target to the end for a clear feature/target boundary.
    target = df.pop("rented_bike_count")
    df["rented_bike_count"] = target

    df.to_csv(PROCESSED_PATH, index=False)

    print("=== Prepared dataset ===")
    print(f"Saved to: {PROCESSED_PATH}")
    print(f"Shape: {df.shape}")
    print("\nColumns and dtypes:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nTarget (rented_bike_count) summary:")
    print(df["rented_bike_count"].describe())


if __name__ == "__main__":
    main()

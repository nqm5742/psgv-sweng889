"""Explore the raw Seoul Bike Sharing dataset.

Read-only inspection: no cleaning, transformation, or modeling is performed here.
"""

import pandas as pd

DATA_PATH = "data/raw/SeoulBikeData.csv"


def main():
    # The header row contains non-UTF-8 characters (degree signs), so use cp1252.
    df = pd.read_csv(DATA_PATH, encoding="cp1252")

    print("=== Shape ===")
    print(df.shape)

    print("\n=== Column names ===")
    print(list(df.columns))

    print("\n=== First 5 rows ===")
    print(df.head())

    print("\n=== Data types ===")
    print(df.dtypes)

    print("\n=== Missing values per column ===")
    print(df.isnull().sum())

    print("\n=== Duplicate rows ===")
    print(df.duplicated().sum())

    print("\n=== Descriptive statistics ===")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()

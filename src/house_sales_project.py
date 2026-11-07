"""House sales project: cleaning, summary statistics, and regression models.

Expected local inputs under data/raw/:
    house_sales.csv
    train.csv
    validation.csv

Source datasets are intentionally not included in version control.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
EXPECTED_COLUMNS = [
    "house_id", "city", "sale_price", "sale_date",
    "months_listed", "bedrooms", "house_type", "area",
]
ALLOWED_CITIES = {"Silvertown", "Riverford", "Teasdale", "Poppleton", "Unknown"}
HOUSE_TYPE_MAP = {
    "Det.": "Detached",
    "Semi": "Semi-detached",
    "Terr.": "Terraced",
}


def read_input_csv(filename: str) -> pd.DataFrame:
    """Read a CSV file and interpret '--' as a missing value."""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Place the CSV file in data/raw/."
        )
    return pd.read_csv(path, na_values=["--"])


def clean_house_sales(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean house_sales data using documented assumptions."""
    df = raw.copy()

    # Remove records without a usable target.
    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")
    df = df.dropna(subset=["sale_price"]).copy()
    df["sale_price"] = df["sale_price"].round().astype("int64")

    # City
    df["city"] = df["city"].fillna("Unknown").astype("string").str.strip()
    df.loc[~df["city"].isin(ALLOWED_CITIES), "city"] = "Unknown"

    # Sale date
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df["sale_date"] = df["sale_date"].fillna(pd.Timestamp("2023-01-01"))

    # Months listed
    df["months_listed"] = pd.to_numeric(df["months_listed"], errors="coerce")
    mean_months = round(float(df["months_listed"].mean()), 1)
    df["months_listed"] = df["months_listed"].fillna(mean_months).round(1)

    # Bedrooms
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
    mean_bedrooms = int(round(float(df["bedrooms"].mean())))
    df["bedrooms"] = df["bedrooms"].fillna(mean_bedrooms).round().astype("int64")

    # House type
    df["house_type"] = (
        df["house_type"].astype("string").str.strip().replace(HOUSE_TYPE_MAP)
    )
    mode = df["house_type"].mode(dropna=True)
    if mode.empty:
        raise ValueError("Cannot impute house_type: no non-missing values.")
    df["house_type"] = df["house_type"].fillna(mode.iloc[0])

    # Area is supplied as text such as '107.8 sq.m.'
    area = (
        df["area"].astype("string")
        .str.replace(" sq.m.", "", regex=False)
        .str.strip()
    )
    df["area"] = pd.to_numeric(area, errors="coerce")
    mean_area = round(float(df["area"].mean()), 1)
    df["area"] = df["area"].fillna(mean_area).round(1)

    cleaned = df[EXPECTED_COLUMNS].copy()
    validate_clean_data(cleaned)
    return cleaned


def validate_clean_data(df: pd.DataFrame) -> None:
    """Fail early if the cleaned data do not meet the expected contract."""
    if list(df.columns) != EXPECTED_COLUMNS:
        raise AssertionError(f"Unexpected columns: {list(df.columns)}")
    if not df["city"].isin(ALLOWED_CITIES).all():
        raise AssertionError("Unexpected city values remain.")
    if not df["house_type"].isin({"Terraced", "Semi-detached", "Detached"}).all():
        raise AssertionError("Unexpected house_type values remain.")
    if df.isna().any().any():
        missing = df.isna().sum()
        raise AssertionError(f"Missing values remain:\n{missing[missing > 0]}")


def summarize_prices_by_bedrooms(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Return mean sale price and sample variance for each bedroom count."""
    summary = (
        cleaned.groupby("bedrooms", as_index=False)
        .agg(
            avg_price=("sale_price", "mean"),
            var_price=("sale_price", "var"),
        )
        .round(1)
    )
    return summary


def prepare_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Normalize feature formats and separate target if it is present."""
    data = df.copy()
    y = None
    if "sale_price" in data.columns:
        y = pd.to_numeric(data["sale_price"], errors="coerce")
        data = data.drop(columns=["sale_price"])

    if "house_id" in data.columns:
        data = data.drop(columns=["house_id"])
    if "sale_date" in data.columns:
        # Deliberately excluded from this baseline/comparison pair.
        data = data.drop(columns=["sale_date"])

    if "area" in data.columns:
        data["area"] = (
            data["area"].astype("string")
            .str.replace(" sq.m.", "", regex=False)
            .str.strip()
        )
        data["area"] = pd.to_numeric(data["area"], errors="coerce")
    if "months_listed" in data.columns:
        data["months_listed"] = pd.to_numeric(data["months_listed"], errors="coerce")
    if "bedrooms" in data.columns:
        data["bedrooms"] = pd.to_numeric(data["bedrooms"], errors="coerce")
    if "house_type" in data.columns:
        data["house_type"] = (
            data["house_type"].astype("string").str.strip().replace(HOUSE_TYPE_MAP)
        )
    if "city" in data.columns:
        data["city"] = data["city"].astype("string").str.strip()

    # Convert pandas nullable extension dtypes to NumPy-compatible object
    # columns before passing data to scikit-learn's imputers.
    numeric_columns = set(data.select_dtypes(include=["number"]).columns)
    for column in data.columns:
        if column not in numeric_columns:
            data[column] = data[column].astype("object").where(
                data[column].notna(), np.nan
            )
    return data, y


def make_model_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    X = X.copy()
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]
    # Normalize pandas nullable-string missing values for scikit-learn imputers.
    for column in categorical_columns:
        X[column] = X[column].astype("object").where(X[column].notna(), np.nan)

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("model", estimator)]
    )


def run_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train baseline and comparison models, then predict validation rows."""
    train_raw = read_input_csv("train.csv")
    valid_raw = read_input_csv("validation.csv")

    if "house_id" not in valid_raw.columns:
        raise ValueError("validation.csv must contain house_id.")
    valid_ids = valid_raw["house_id"].copy()

    X_train, y_train = prepare_model_features(train_raw)
    X_valid, _ = prepare_model_features(valid_raw)

    if y_train is None:
        raise ValueError("train.csv must contain sale_price.")
    usable = y_train.notna()
    X_train = X_train.loc[usable].copy()
    y_train = y_train.loc[usable].astype(float)

    # Align feature columns without fitting preprocessing on validation rows.
    X_valid = X_valid.reindex(columns=X_train.columns)

    models = [
        ("baseline", LinearRegression()),
        ("comparison", RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )),
    ]
    outputs = []
    for label, estimator in models:
        pipeline = make_model_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_valid)
        result = pd.DataFrame({"house_id": valid_ids, "price": predictions})
        outputs.append(result)
        result.to_csv(OUTPUT_DIR / f"{label}_predictions.csv", index=False)
        print(f"{label}: wrote {len(result)} validation predictions")

    return outputs[0], outputs[1]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_sales = read_input_csv("house_sales.csv")
    missing_city = int(raw_sales["city"].isna().sum())
    print(f"Missing city values in source: {missing_city}")

    cleaned = clean_house_sales(raw_sales)
    cleaned.to_csv(OUTPUT_DIR / "clean_house_sales.csv", index=False)

    summary = summarize_prices_by_bedrooms(cleaned)
    summary.to_csv(OUTPUT_DIR / "price_by_bedrooms.csv", index=False)

    print(f"Cleaned data shape: {cleaned.shape}")
    print(f"Missing values after cleaning: {int(cleaned.isna().sum().sum())}")
    print("\nAverage price and variance by bedrooms:")
    print(summary.to_string(index=False))

    run_models()
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()



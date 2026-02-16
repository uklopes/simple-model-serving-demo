import json
import pathlib
import pickle
from typing import List
from typing import Tuple
from joblib import dump as joblib_dump

import lightgbm
import pandas
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
     'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
]

# Simplified model: only core house attributes (no zipcode, no demographics).
SIMPLE_SALES_COLUMN_SELECTION = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",  # used only to join demographics; dropped before training
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv("data/zipcode_demographics.csv",
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def load_data_simple(
    sales_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load simplified house features and join demographics (via zipcode)."""
    data = pandas.read_csv(
        sales_path,
        usecols=sales_column_selection,
        dtype={"zipcode": str},
    )
    demographics = pandas.read_csv(
        "data/zipcode_demographics.csv",
        dtype={"zipcode": str},
    )

    merged_data = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    y = merged_data.pop("price")
    x = merged_data
    return x, y


def main():
    """Load data, train model, and export artifacts."""
    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # ---- Full model (includes demographics join) ----
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=42
    )

    model = pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        lightgbm.LGBMRegressor()
    ).fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mape = (abs((y_test - y_pred) / y_test)).mean() * 100
    rmse = (metrics.mean_squared_error(y_test, y_pred) ** 0.5)
    print("Full Model Performance Metrics (Test Dataset):")
    print(f"  R² Score: {metrics.r2_score(y_test, y_pred):.4f}")
    print(f"  MAE: ${metrics.mean_absolute_error(y_test, y_pred):,.0f}")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  MAPE: {mape:.2f}%")

    # Output model artifacts: use joblib for better LightGBM compatibility
    joblib_dump(model, output_dir / "model.pkl")
    json.dump(list(x_train.columns), open(output_dir / "model_features.json", "w"))

    # ---- Simplified model (no demographics) ----
    x_s, y_s = load_data_simple(SALES_PATH, SIMPLE_SALES_COLUMN_SELECTION)
    x_s_train, x_s_test, y_s_train, y_s_test = model_selection.train_test_split(
        x_s, y_s, random_state=42
    )

    simple_model = pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        lightgbm.LGBMRegressor()
    ).fit(x_s_train, y_s_train)

    y_s_pred = simple_model.predict(x_s_test)
    s_mape = (abs((y_s_test - y_s_pred) / y_s_test)).mean() * 100
    s_rmse = (metrics.mean_squared_error(y_s_test, y_s_pred) ** 0.5)
    print("\nSimplified Model Performance Metrics (Test Dataset):")
    print(f"  R² Score: {metrics.r2_score(y_s_test, y_s_pred):.4f}")
    print(f"  MAE: ${metrics.mean_absolute_error(y_s_test, y_s_pred):,.0f}")
    print(f"  RMSE: ${s_rmse:,.0f}")
    print(f"  MAPE: {s_mape:.2f}%")

    joblib_dump(simple_model, output_dir / "model_simple.pkl")
    json.dump(list(x_s_train.columns), open(output_dir / "model_simple_features.json", "w"))


if __name__ == "__main__":
    main()
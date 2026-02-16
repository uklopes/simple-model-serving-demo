"""
Prediction-specific domain logic.

This is intentionally separate from `utils/__init__.py` to keep the package
initializer from becoming a "god module".
"""

from typing import Dict, Any

import csv
import json
import pathlib
from functools import lru_cache

import numpy as np

from . import get_model, load_feature_names, logger


DATA_DIR = pathlib.Path("data")
DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"
MODEL_DIR = pathlib.Path("model")
SIMPLE_FEATURES_PATH = MODEL_DIR / "model_simple_features.json"


@lru_cache(maxsize=1)
def load_demographics() -> Dict[int, Dict[str, Any]]:
    """Load demographic data from CSV."""
    if not DEMOGRAPHICS_PATH.exists():
        logger.error(f"Demographics file not found at {DEMOGRAPHICS_PATH}")
        return {}

    demog_data: Dict[int, Dict[str, Any]] = {}
    try:
        with open(DEMOGRAPHICS_PATH, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                zipcode_str = row.get("zipcode")
                if zipcode_str:
                    try:
                        zipcode = int(zipcode_str)
                        processed_row = {k: float(v) for k, v in row.items() if k != "zipcode"}
                        demog_data[zipcode] = processed_row
                    except ValueError:
                        continue
        logger.info(f"Loaded demographic data for {len(demog_data)} zipcodes")
    except Exception as e:
        logger.error(f"Error loading demographics: {e}")
        demog_data = {}
    return demog_data


def join_demographics(house_features: Dict[str, Any]) -> Dict[str, Any]:
    """Join house features with demographic data."""
    zipcode_val = house_features.get("zipcode")
    if zipcode_val is None:
        return house_features
    try:
        zipcode = int(str(zipcode_val))
    except (ValueError, TypeError):
        return house_features

    demographics = load_demographics()
    if zipcode in demographics:
        combined = house_features.copy()
        combined.update(demographics[zipcode])
        return combined

    logger.warning(f"No demographic data found for zipcode {zipcode}")
    return house_features


def predict_house_price(features: Dict[str, Any], model_name: str | None = None) -> float:
    """Core prediction logic including demographic join."""
    resolved_model_name = model_name or None
    model = get_model(resolved_model_name) if resolved_model_name else get_model()
    model_feature_names = load_simple_feature_names() if resolved_model_name == "simple" else load_feature_names()

    if model is None or not model_feature_names:
        raise RuntimeError("Model or feature names not loaded")

    # 1. Join demographics on the backend
    features_with_demographics = join_demographics(features)

    # 2. Prepare feature array in the correct order for the model
    feature_array = []
    for feature_name in model_feature_names:
        value = features_with_demographics.get(feature_name)
        if value is None:
            raise ValueError(f"Feature {feature_name} missing after demographic join")
        feature_array.append(float(value))

    # Convert to numpy array and reshape for prediction
    feature_array = np.array(feature_array).reshape(1, -1)

    # Make prediction
    try:
        prediction = model.predict(feature_array)[0]
        return float(prediction)
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise RuntimeError(f"Error making prediction: {e}")


@lru_cache(maxsize=1)
def load_simple_feature_names() -> list[str]:
    """Load feature names for the simplified model."""
    try:
        logger.info(f"Loading simple feature names from {SIMPLE_FEATURES_PATH}")
        with open(SIMPLE_FEATURES_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            features = data.get("features", [])
            return [str(x) for x in features] if isinstance(features, list) else []
        return []
    except Exception as e:
        logger.error(f"Error loading simple feature names: {e}")
        return []


#
# NOTE: `predict_house_price_simple` was intentionally removed in favor of
# `predict_house_price(..., model_name="simple")`.

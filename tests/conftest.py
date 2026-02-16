"""
Pytest configuration and shared fixtures.
"""
import pytest
import json
import pathlib
import pickle
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

# Import the app
from main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock sklearn model for testing."""
    model = LinearRegression()
    # Train on dummy data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([10, 20, 30])
    model.fit(X, y)
    return model


@pytest.fixture
def sample_feature_names():
    """Sample feature names for testing."""
    return [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
        "sqft_above", "sqft_basement", "ppltn_qty", "urbn_ppltn_qty",
        "sbrbn_ppltn_qty", "farm_ppltn_qty", "non_farm_qty",
        "medn_hshld_incm_amt", "medn_incm_per_prsn_amt", "hous_val_amt",
        "edctn_less_than_9_qty", "edctn_9_12_qty", "edctn_high_schl_qty",
        "edctn_some_clg_qty", "edctn_assoc_dgre_qty", "edctn_bchlr_dgre_qty",
        "edctn_prfsnl_qty", "per_urbn", "per_sbrbn", "per_farm",
        "per_non_farm", "per_less_than_9", "per_9_to_12", "per_hsd",
        "per_some_clg", "per_assoc", "per_bchlr", "per_prfsnl"
    ]


@pytest.fixture
def sample_features_dict(sample_feature_names):
    """Sample features dictionary for testing."""
    features = {
        name: 1.0 if name.startswith('per_') else 100.0
        for name in sample_feature_names
    }
    # Add zipcode for demographic join
    features["zipcode"] = 98118
    return features


@pytest.fixture
def sample_features_list(sample_feature_names):
    """Sample features as a list for testing."""
    return [1.0 if name.startswith('per_') else 100.0 for name in sample_feature_names]


@pytest.fixture
def sample_house_request():
    """Sample house request matching PredictionRequest format."""
    return {
        "bedrooms": 3.0,
        "bathrooms": 2.5,
        "sqft_living": 2000.0,
        "sqft_lot": 5000.0,
        "floors": 1.5,
        "waterfront": 0.0,
        "view": 0.0,
        "condition": 3.0,
        "grade": 7.0,
        "sqft_above": 1500.0,
        "sqft_basement": 500.0,
        "yr_built": 1980.0,
        "yr_renovated": 0.0,
        "zipcode": 98118,
        "lat": 47.5,
        "long": -122.3,
        "sqft_living15": 1800.0,
        "sqft_lot15": 5500.0
    }


@pytest.fixture
def mock_redis():
    """Mock Redis connection."""
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    return mock_redis


@pytest.fixture
def mock_queue():
    """Mock RQ Queue."""
    mock_queue = MagicMock()
    mock_job = MagicMock()
    mock_job.id = "test-job-id-123"
    mock_queue.enqueue.return_value = mock_job
    return mock_queue

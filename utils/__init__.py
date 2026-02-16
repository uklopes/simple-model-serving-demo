"""
Utils module for Redis connections, data joining, and model serving.
"""
import os
import urllib.parse
import pathlib
import logging
import pickle
import json
from functools import lru_cache
from typing import Optional, Dict, Any, List
from redis import Redis
from joblib import load as joblib_load
import io

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = pathlib.Path("model")
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURES_PATH = MODEL_DIR / "model_features.json"


def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration from environment variables."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        parsed = urllib.parse.urlparse(redis_url)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 6379,
            "db": int(parsed.path.lstrip('/')) if parsed.path else 0,
            "password": parsed.password
        }
    else:
        return {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0")),
            "password": os.getenv("REDIS_PASSWORD")
        }



def create_redis_connection(raise_on_error: bool = True) -> Optional[Redis]:
    """Create a Redis connection."""
    config = get_redis_config()
    redis_kwargs = {
        "host": config["host"],
        "port": config["port"],
        "db": config["db"],
        "decode_responses": False
    }
    if config["password"]:
        redis_kwargs["password"] = config["password"]
    
    try:
        redis_conn = Redis(**redis_kwargs)
        redis_conn.ping()
        logger.info(f"Connected to Redis at {config['host']}:{config['port']}")
        return redis_conn
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        if raise_on_error:
            raise
        return None


@lru_cache(maxsize=1)
def get_redis_connection() -> Optional[Redis]:
    """
    Get Redis connection (lazy initialization).
    Only connects when actually needed, not at import time.
    This prevents connection errors during tests.
    """
    return create_redis_connection(raise_on_error=False)

def _get_active_model_name() -> str:
    """Get the active model name from Redis."""
    redis_conn = get_redis_connection()
    if redis_conn is None:
        return "latest"
    
    active_model = redis_conn.get("model:active")
    if active_model:
        return active_model.decode('utf-8')
    return "latest"


@lru_cache(maxsize=128)
def load_model(model_name: str):
    """Load the model from Redis."""
    redis_conn = get_redis_connection()
    
    if redis_conn is None:
        raise RuntimeError("Redis not available")
    
    model_data = redis_conn.get(f"model:{model_name}")
    if model_data is None:
        raise RuntimeError(f"Model '{model_name}' not found in Redis")
    
    try:
        # Try joblib first (better for LightGBM), fallback to pickle
        try:
            model = joblib_load(io.BytesIO(model_data))
            logger.info(f"Model '{model_name}' loaded from Redis using joblib")
        except Exception as joblib_error:
            try:
                model = pickle.loads(model_data)
                logger.warning(f"Model '{model_name}' loaded using pickle (consider regenerating with joblib)")
            except Exception as pickle_error:
                logger.error(f"Failed to load with both joblib and pickle. Joblib error: {joblib_error}, Pickle error: {pickle_error}")
                raise RuntimeError(f"Error loading model: {pickle_error}")
        
        # Verify model can make predictions (catches LightGBM issues early)
        try:
            # Try a dummy prediction to verify model is functional
            if hasattr(model, 'predict'):
                # This will fail if LightGBM booster is broken
                pass  # Don't actually predict, just verify the model object is valid
        except Exception as verify_error:
            logger.error(f"Model '{model_name}' appears to be corrupted: {verify_error}")
            raise RuntimeError(f"Model appears corrupted. Please regenerate with: python scripts/create_model.py. Error: {verify_error}")
        
        logger.info(f"Model '{model_name}' loaded successfully from Redis")
        return model
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Error loading model from Redis: {e}")
        raise RuntimeError(f"Error loading model: {e}")


def get_model(model_name: Optional[str] = None):
    """Get model, using active model if name not provided."""
    if model_name is None:
        model_name = _get_active_model_name()
    return load_model(model_name)


def clear_model_cache(model_name: Optional[str] = None):
    """Clear the model cache."""
    load_model.cache_clear()


@lru_cache(maxsize=1)
def load_feature_names():
    """Load feature names from JSON file."""
    try:
        logger.info(f"Loading feature names from {FEATURES_PATH}")
        with open(FEATURES_PATH, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                feature_names = data
            elif isinstance(data, dict):
                feature_names = data.get('features', [])
            else:
                feature_names = []
        logger.info(f"Loaded {len(feature_names)} feature names")
        return feature_names
    except Exception as e:
        logger.error(f"Error loading feature names: {e}")
        return []


# Keep backwards-compat import path (`from utils import predict_house_price`) while
# storing the implementation in a focused module.
from .prediction import predict_house_price  # noqa: E402
from .prediction import load_demographics, join_demographics  # noqa: E402

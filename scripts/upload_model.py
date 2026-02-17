#!/usr/bin/env python3
import logging
import requests
import time
import sys
import pathlib

api_url = "http://localhost:8000"

# Logging (stdout-friendly for Docker/Railway)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("upload_model")

# Model names as stored in Redis
full_model_name = "default"
simple_model_name = "simple"

full_model_path = pathlib.Path("model/model.pkl")
simple_model_path = pathlib.Path("model/model_simple.pkl")

# Optional: allow overriding the API URL (e.g. docker host/port)
if len(sys.argv) > 1:
    api_url = sys.argv[1].rstrip("/")
logger.info(f"Using API url: {api_url}")

# Wait for API to be ready
for _ in range(30):
    try:
        r = requests.get(f"{api_url}/health", timeout=2)
        logger.info(f"API health: {r.status_code} {r.text}")
        break
    except Exception as e:
        logger.info(f"Waiting for API to be ready... ({e})")
        time.sleep(1)
else:
    logger.error("Timed out waiting for API /health")
    sys.exit(1)

# Check registered models
try:
    response = requests.get(f"{api_url}/model/list", timeout=10)
    response.raise_for_status()
    data = response.json()
except Exception as e:
    logger.error(f"Failed to fetch /model/list: {e}")
    sys.exit(1)

models = data.get("models", [])
active = data.get("active")
logger.info(f"Models in Redis: {models}")
logger.info(f"Active model: {active}")

def _register_model_if_missing(model_path: pathlib.Path, model_name: str) -> None:
    if not model_path.exists():
        logger.warning(f"Model file not found; skipping register: {model_path}")
        return
    if model_name in models:
        logger.info(f"Model already registered; skipping: {model_name}")
        return
    with open(model_path, "rb") as f:
        logger.info(f"Registering model '{model_name}' from {model_path}...")
        res = requests.post(
            f"{api_url}/model/register",
            files={"model_file": f},
            data={"model_name": model_name},
            timeout=30,
        )
        if res.ok:
            logger.info(f"Registered model '{model_name}': {res.status_code} {res.text}")
        else:
            logger.error(f"Failed to register model '{model_name}': {res.status_code} {res.text}")


# Always try to register missing models (does not change active model)
_register_model_if_missing(full_model_path, full_model_name)
_register_model_if_missing(simple_model_path, simple_model_name)

# If there is no active model, set the full model as active (if present)
if not active:
    if full_model_name in models or full_model_path.exists():
        logger.info(f"Setting active model to '{full_model_name}'...")
        res = requests.post(
            f"{api_url}/model/set-active",
            params={"model_name": full_model_name},
            timeout=10,
        )
        if res.ok:
            logger.info(f"Active model set: {res.status_code} {res.text}")
        else:
            logger.error(f"Failed to set active model: {res.status_code} {res.text}")
else:
    logger.info("Active model already set; not changing it.")

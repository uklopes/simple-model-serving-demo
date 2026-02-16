#!/usr/bin/env python3
import requests
import time
import sys
import pathlib

api_url = "http://localhost:8000"

# Model names as stored in Redis
full_model_name = "default"
simple_model_name = "simple"

full_model_path = pathlib.Path("model/model.pkl")
simple_model_path = pathlib.Path("model/model_simple.pkl")

# Optional: allow overriding the API URL (e.g. docker host/port)
if len(sys.argv) > 1:
    api_url = sys.argv[1].rstrip("/")

# Wait for API to be ready
for _ in range(30):
    try:
        requests.get(f"{api_url}/health", timeout=2)
        break
    except:
        time.sleep(1)
else:
    sys.exit(1)

# Check registered models
response = requests.get(f"{api_url}/model/list").json()
models = response.get("models", [])
active = response.get("active")

def _register_model_if_missing(model_path: pathlib.Path, model_name: str) -> None:
    if not model_path.exists():
        return
    if model_name in models:
        return
    with open(model_path, "rb") as f:
        requests.post(
            f"{api_url}/model/register",
            files={"model_file": f},
            data={"model_name": model_name},
            timeout=30,
        )


# Always try to register missing models (does not change active model)
_register_model_if_missing(full_model_path, full_model_name)
_register_model_if_missing(simple_model_path, simple_model_name)

# If there is no active model, set the full model as active (if present)
if not active:
    if full_model_name in models or full_model_path.exists():
        requests.post(f"{api_url}/model/set-active", params={"model_name": full_model_name})

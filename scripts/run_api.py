__test__ = False

import requests
import pandas as pd
import numpy as np
import time
import sys

# Configuration
BASE_URL = "http://localhost:8000"
SYNC_ENDPOINT = f"{BASE_URL}/predict/sync"
SIMPLE_ENDPOINT = f"{BASE_URL}/predict/simple"
ASYNC_ENDPOINT = f"{BASE_URL}/predict"
JOB_STATUS_ENDPOINT = f"{BASE_URL}/predict/job"
DATA_PATH = "data/future_unseen_examples.csv"

def test_health():
    print("\n--- Testing Health Check ---")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def _ensure_zipcode_int(payload: dict) -> dict:
    """Ensure zipcode is an int (pandas may parse as float, e.g. 98118.0)."""
    if "zipcode" in payload and payload["zipcode"] is not None and not pd.isna(payload["zipcode"]):
        payload["zipcode"] = int(payload["zipcode"])
    return payload


def _print_stats(label: str, values: list[float]) -> None:
    if not values:
        print(f"{label}: no values")
        return
    arr = np.array(values, dtype=float)
    print(f"{label}: n={len(values)}")
    print(f"  mean   : {arr.mean():.6f}")
    print(f"  median : {np.median(arr):.6f}")
    print(f"  std    : {arr.std(ddof=0):.6f}")


def run_sync_predictions(examples) -> list[float]:
    print("\n--- Running Synchronous Predictions (all rows) ---")
    predictions: list[float] = []
    for i, example in examples.iterrows():
        payload = _ensure_zipcode_int(example.to_dict())

        # Keep output readable: print progress every 10 rows
        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == len(examples):
            print(f"Sync progress: {i+1}/{len(examples)}")

        response = requests.post(SYNC_ENDPOINT, json=payload)
        if response.status_code != 200:
            print(f"Sync error on row {i+1}: status={response.status_code} body={response.text}")
            continue
        predictions.append(float(response.json()["prediction"]))
    return predictions


def run_simple_predictions(examples) -> list[float]:
    print("\n--- Running Simple Predictions (all rows) ---")
    predictions: list[float] = []
    simple_fields = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
        "zipcode",
    ]

    for i, example in examples.iterrows():
        full_payload = _ensure_zipcode_int(example.to_dict())
        payload = {k: full_payload[k] for k in simple_fields}

        # Keep output readable: print progress every 10 rows
        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == len(examples):
            print(f"Simple progress: {i+1}/{len(examples)}")

        response = requests.post(SIMPLE_ENDPOINT, json=payload)
        if response.status_code != 200:
            print(f"Simple error on row {i+1}: status={response.status_code} body={response.text}")
            continue

        data = response.json()
        if not isinstance(data, dict) or "prediction" not in data:
            print(f"Simple response missing 'prediction' on row {i+1}: {data}")
            continue
        predictions.append(float(data["prediction"]))
    return predictions


def run_async_predictions(examples, poll_interval_s: float = 1.0, timeout_s: float = 120.0) -> list[float]:
    print("\n--- Running Asynchronous Predictions (all rows) ---")

    # 1) Enqueue all jobs
    job_ids: list[str] = []
    for i, example in examples.iterrows():
        payload = _ensure_zipcode_int(example.to_dict())

        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == len(examples):
            print(f"Async enqueue progress: {i+1}/{len(examples)}")

        response = requests.post(ASYNC_ENDPOINT, json=payload)
        if response.status_code != 200:
            print(f"Async enqueue error on row {i+1}: status={response.status_code} body={response.text}")
            continue
        job_ids.append(response.json()["job_id"])

    # 2) Poll until all are done (or timeout)
    pending = set(job_ids)
    predictions: list[float] = []
    start = time.time()

    while pending and (time.time() - start) < timeout_s:
        time.sleep(poll_interval_s)

        finished_this_round: list[str] = []
        for job_id in list(pending):
            status_url = f"{JOB_STATUS_ENDPOINT}/{job_id}"
            res = requests.get(status_url)
            if res.status_code != 200:
                # keep it pending; endpoint can return transient errors
                continue
            status_data = res.json()
            status = status_data.get("status")

            if status == "finished":
                result = status_data.get("result")
                
                if isinstance(result, dict):
                    pred = result.get("prediction")
                else:
                    pred = result
                if pred is not None:
                    predictions.append(float(pred))
                finished_this_round.append(job_id)
            elif status == "failed":
                print(f"Async job failed: {job_id} error={status_data.get('error')}")
                finished_this_round.append(job_id)

        for job_id in finished_this_round:
            pending.discard(job_id)

        if pending:
            print(f"Async polling: {len(job_ids) - len(pending)}/{len(job_ids)} finished")

    if pending:
        print(f"Async polling timed out with {len(pending)} jobs still pending.")

    return predictions

if __name__ == "__main__":
    # Check if a URL was provided as argument
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1].rstrip('/')
        SYNC_ENDPOINT = f"{BASE_URL}/predict/sync"
        SIMPLE_ENDPOINT = f"{BASE_URL}/predict/simple"
        ASYNC_ENDPOINT = f"{BASE_URL}/predict"
        JOB_STATUS_ENDPOINT = f"{BASE_URL}/predict/job"
        print(f"Using remote URL: {BASE_URL}")

    try:
        df = pd.read_csv(DATA_PATH, dtype={"zipcode": "int64"})
        print(f"Loaded {len(df)} examples from {DATA_PATH}")
        
        test_health()
        sync_preds = run_sync_predictions(df)
        simple_preds = run_simple_predictions(df)
        async_preds = run_async_predictions(df)

        print("\n--- Summary Statistics ---")
        _print_stats("sync", sync_preds)
        _print_stats("simple", simple_preds)
        _print_stats("async", async_preds)
        
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found. Please ensure the data folder is present.")
    except Exception as e:
        print(f"Unexpected error: {e}")

### Simple Model Service Demo

A small FastAPI service that serves **house price predictions** with:
- a **synchronous** endpoint for full-feature payloads
- an **async** endpoint backed by **RQ + Redis**
- a **bonus “simple” synchronous** endpoint that uses a **separately-trained simplified model**

---

### What’s implemented

- **FastAPI prediction API**
  - **`POST /predict/sync`**: synchronous prediction (returns `{"prediction": ...}`)
  - **`POST /predict`**: async prediction (enqueues an RQ job)
  - **`GET /predict/job/{job_id}`**: poll job status + result
  - **`POST /predict/simple`**: synchronous “simple” prediction that calls the simplified model (`model_name="simple"`)

- **RQ worker + Redis queue**
  - **Queue name**: `predictions`
  - **Worker task**: `worker.predict_house_price_task`

- **Model registry stored in Redis**
  - **`POST /model/register`**: upload a model artifact into Redis
  - **`POST /model/set-active`**: choose the “active” model used by the full endpoints
  - **`GET /model/list`**: list registered models and current active model

- **Two model variants**
  - **Full model**: trained on a richer set of house attributes + joined zipcode demographics.
  - **Simplified model**: trained on a reduced set of house attributes **but still joins zipcode demographics**.
  - Artifacts written to `model/`:
    - **Full**: `model.pkl`, `model_features.json`
    - **Simple**: `model_simple.pkl`, `model_simple_features.json`

- **Backend demographics join**
  - The backend joins `zipcode` → demographics from `data/zipcode_demographics.csv` during prediction.
  - If a zipcode is missing from the CSV, the join is skipped for that request and prediction may fail if required demographic features are missing.

- **Client test / demo script**
  - `scripts/run_api.py` loads `data/future_unseen_examples.csv` and runs:
    - full sync predictions
    - simple sync predictions
    - full async predictions (enqueue + poll)
  - Prints mean/median/std of each set.

- **Dependency hardening**
  - **Pinned `lightgbm==3.3.5`** to avoid deserialization/predict incompatibilities across environments.

---

### Quickstart

#### Local (python)

1) Install deps:

```bash
pip install -r requirements.txt
```

2) Train artifacts:

```bash
python scripts/create_model.py
```

3) Start Redis (example):

```bash
docker run -p 6379:6379 redis:7-alpine
```

4) Start API + worker (in separate shells):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
python worker.py
```

5) Upload models (stores in Redis):

```bash
python scripts/upload_model.py http://localhost:8000
```

6) Run the demo:

```bash
python scripts/run_api.py http://localhost:8000
```

#### Docker Compose

```bash
docker-compose up -d --build
python scripts/run_api.py http://localhost:8000
```

---

### Design notes (high level)

- **Sync vs async**
  - `/predict/sync` and `/predict/simple` compute immediately.
  - `/predict` uses RQ so the API can return quickly and work can be scaled with more workers.

- **Why a separate “simple” model**
  - The full model expects many house columns; the simple endpoint accepts fewer.
  - The clean way to support that is to **train a model that truly matches the minimal input schema**, rather than guessing defaults for missing features.

---

### Future improvements

- **Better error handling**
  - Return `400` for “missing features after demographic join” instead of `500`.
  - Clear error when `zipcode` isn’t found in demographics.

- **Cleaner module boundaries**
  - Split `utils/__init__.py` into dedicated modules (`redis.py`, `models.py`, `features.py`) instead of a package initializer exporting everything.

- **Pydantic v2 updates**
  - Replace deprecated `request.dict()` with `request.model_dump()`.

- **Operational hardening**
  - Health checks for Redis connectivity and worker readiness.
  - Metrics/tracing for job latency + failure rate.

- **Performance**
  - Batch prediction endpoint.
  - Avoid per-request feature-name loading; cache and version feature lists explicitly.

- **Model lifecycle**
  - Stronger model versioning/metadata in Redis (feature schema hash, training time, metrics).
  - Safer “active model” rollouts (canary / gradual switching).

